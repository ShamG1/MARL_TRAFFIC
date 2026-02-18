#include "ScenarioEnv.h"
#ifdef CPP_MCTS_ENABLE_RENDER
#include "Renderer.h"
#endif
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

static constexpr float PI_F = 3.14159265358979323846f;

static inline float wrap_angle_rad(float a) {
    a = std::fmod(a + PI_F, 2.0f * PI_F);
    if (a < 0) a += 2.0f * PI_F;
    return a - PI_F;
}

static inline float compute_progress(Car &car, const RewardConfig& cfg) {
    // Path-based progress: use path_index as primary progress signal.
    // This reduces corner-cutting towards the final goal point and
    // aligns shaping with the route topology.
    if (car.path.empty()) return 0.0f;

    float r = 0.0f;
    // Use path_index as a monotonic progress measure in [0, path.size()-1]
    float cur_idx = float(car.path_index);
    if (car.prev_dist_to_goal > -0.5f) { // repurpose prev_dist_to_goal as prev_index storage
        float delta = cur_idx - car.prev_dist_to_goal;
        if (delta > 0.0f) {
            // Normalize by path length to keep scale roughly invariant
            float max_idx = std::max(1.0f, float((int)car.path.size() - 1));
            float normalized = delta / max_idx;
            r = cfg.k_prog * normalized;
        }
    }
    // store current index in prev_dist_to_goal to avoid extra member
    car.prev_dist_to_goal = cur_idx;
    return r;
}

static inline float compute_stuck(const Car &car, const RewardConfig &cfg) {
    float speed_ms = (car.state.v * FPS) / SCALE;
    if (speed_ms >= cfg.v_min_ms) return 0.0f;
    // Continuous shaping: penalize proportional to how far below v_min we are.
    float diff = cfg.v_min_ms - speed_ms; // >= 0
    // A small scaling keeps magnitude comparable to the old constant penalty.
    // For example, diff in [0, 1] -> penalty in approximately [0, |k_stuck|].
    return cfg.k_stuck * diff;
}

static inline float compute_smooth(Car &car, const RewardConfig &cfg) {
    float current_acc_norm = car.acc / MAX_ACC;
    float current_steer_norm = car.steering_angle / MAX_STEERING_ANGLE;

    float d0 = current_acc_norm - car.prev_action.first;
    float d1 = current_steer_norm - car.prev_action.second;
    float diff2 = d0 * d0 + d1 * d1; // squared norm
    float r = cfg.k_sm * diff2;

    car.prev_action = {current_acc_norm, current_steer_norm};
    return r;
}

ScenarioEnv::~ScenarioEnv() = default;

void ScenarioEnv::set_route_intents(const std::vector<std::pair<std::pair<std::string,std::string>, int>>& items) {
    route_intents.clear();
    for (const auto& it : items) {
        route_intents[it.first] = it.second;
    }
}

bool ScenarioEnv::load_scenario_bitmaps(const std::string& drivable_png,
                                       const std::string& yellowline_png,
                                       const std::string& dash_png,
                                       const std::string& lane_id_png) {
    BitmapMask road;
    BitmapMask line;
    BitmapMask dash;
    BitmapMask lane;

    if (!road.load_grayscale_png(drivable_png)) return false;
    if (!line.load_grayscale_png(yellowline_png)) return false;
    if (!dash.load_grayscale_png(dash_png)) return false;
    if (!lane.load_grayscale_png(lane_id_png)) return false;

    if (road.width != WIDTH || road.height != HEIGHT) return false;
    if (line.width != WIDTH || line.height != HEIGHT) return false;
    if (dash.width != WIDTH || dash.height != HEIGHT) return false;
    if (lane.width != WIDTH || lane.height != HEIGHT) return false;

    bitmap_road = std::move(road);
    bitmap_line = std::move(line);
    bitmap_dash = std::move(dash);
    bitmap_lane = std::move(lane);
    use_bitmap_scenario = true;

    // Precompute SDF for fast LiDAR raycasting
    bitmap_road.compute_sdf();

    // Automatically rebuild lane layout for the loaded scenario.
    if (scenario_name.find("roundabout") != std::string::npos) {
        lane_layout = build_lane_layout_roundabout_cpp(num_lanes);
    } else if (scenario_name.find("T_") == 0 || scenario_name.find("t_") == 0) {
        lane_layout = build_lane_layout_t_junction_cpp(num_lanes);
    } else if (scenario_name.find("highway") != std::string::npos) {
        lane_layout = build_lane_layout_highway_cpp(num_lanes);
    } else if (scenario_name.find("bottleneck") != std::string::npos) {
        lane_layout = build_lane_layout_bottleneck_cpp(num_lanes);
    } else if (scenario_name.find("merge") != std::string::npos) {
        lane_layout = build_lane_layout_merge_cpp(num_lanes);
    } else {
        lane_layout = build_lane_layout_cpp(num_lanes);
    }
    
    init_traffic_routes();
    return true;
}

void ScenarioEnv::configure(bool use_team, bool respawn, int max_s) {
    use_team_reward = use_team;
    respawn_enabled = respawn;
    max_steps = max_s;
}

void ScenarioEnv::configure_traffic(bool enabled, float density) {
    traffic_flow = enabled;
    traffic_density = density;
    if (traffic_density < 0.0f) traffic_density = 0.0f;
}

void ScenarioEnv::set_traffic_mode(const std::string& mode, int kmax) {
    if (mode == "constant") {
        traffic_mode = TrafficMode::CONSTANT;
    } else {
        traffic_mode = TrafficMode::STOCHASTIC;
    }

    if (kmax < 0) kmax = 0;
    traffic_kmax = kmax;
}

void ScenarioEnv::freeze_traffic(bool freeze) {
    traffic_freeze = freeze;
}

void ScenarioEnv::configure_routes(const std::vector<std::pair<std::string, std::string>>& routes) {
    traffic_routes = routes;
}

void ScenarioEnv::reset() {
    cars.clear();
    lidars.clear();
    agent_ids.clear();

    traffic_cars.clear();
    traffic_lidars.clear();

    next_agent_id = 1;
    step_count = 0;
}

void ScenarioEnv::add_car_with_route(const std::string& start_id, const std::string& end_id) {
    auto it = lane_layout.points.find(start_id);
    if (it == lane_layout.points.end()) {
        return;
    }
    auto spawn = it->second;
    int intent = INTENT_LEFT;
    auto it_int = route_intents.find({start_id, end_id});
    if (it_int != route_intents.end()) {
        intent = it_int->second;
    } else {
        intent = determine_intent(lane_layout, start_id, end_id);
    }
    std::vector<std::pair<float,float>> path;
    if (scenario_name.find("roundabout") != std::string::npos) {
        path = generate_path_roundabout_cpp(lane_layout, num_lanes, intent, start_id, end_id);
    } else if (scenario_name.find("bottleneck") != std::string::npos) {
        path = generate_path_bottleneck_cpp(lane_layout, num_lanes, start_id, end_id);
    } else {
        path = generate_path_cpp(lane_layout, num_lanes, intent, start_id, end_id);
    }

    float heading = 0.0f;
    if (path.size() >= 2) {
        float dx = path[1].first - path[0].first;
        float dy = path[1].second - path[0].second;
        heading = std::atan2(-dy, dx);

        // Spawn deconfliction: if the start position is occupied, shift forward along the path.
        // Use ~1.5 car lengths as the step distance.
        const float norm = std::sqrt(dx * dx + dy * dy);
        if (norm > 1e-6f) {
            const float ux = dx / norm;
            const float uy = dy / norm;

            const float car_len = 54.0f; // Car::length default
            const float min_dist = 1.5f * car_len;
            const int max_tries = 20;

            for (int t = 0; t < max_tries; ++t) {
                bool occupied = false;
                for (const auto& other : cars) {
                    const float ox = other.state.x - spawn.first;
                    const float oy = other.state.y - spawn.second;
                    if ((ox * ox + oy * oy) < (min_dist * min_dist)) {
                        occupied = true;
                        break;
                    }
                }
                if (!occupied) break;
                spawn.first += ux * min_dist;
                spawn.second += uy * min_dist;
            }
        }
    }

    Car c;
    c.state.x = spawn.first;
    c.state.y = spawn.second;
    c.state.v = 0.0f;
    c.state.heading = heading;
    c.spawn_state = c.state;
    c.alive = true;

    c.intention = intent;
    c.path = std::move(path);
    c.path_index = 0;

    c.prev_dist_to_goal = 0.0f;
    c.prev_action = {0.0f, 0.0f};

    cars.push_back(std::move(c));

    // Match Scenario/config.py defaults: LIDAR_RAYS=72, LIDAR_RANGE=250, LIDAR_FOV=360, step=4
    Lidar lid;
    lid.rays = 96;
    lid.fov_deg = 360.0f;
    lid.max_dist = 250.0f;
    lid.step_size = 4.0f;
    lid.distances.assign(lid.rays, lid.max_dist);
    lid.rel_angles.clear();
    {
        const float start_angle_deg = -lid.fov_deg * 0.5f;
        const float step_deg = (lid.rays > 1) ? (lid.fov_deg / float(lid.rays - 1)) : 0.0f;
        constexpr float PI_F2 = 3.14159265358979323846f;
        for (int ii = 0; ii < lid.rays; ++ii) {
            float deg = start_angle_deg + ii * step_deg;
            lid.rel_angles.push_back(deg * PI_F2 / 180.0f);
        }
    }
    lidars.push_back(std::move(lid));

    agent_ids.push_back(next_agent_id++);
}

StepResult ScenarioEnv::step(const std::vector<float>& throttles,
                                const std::vector<float>& steerings,
                                float dt) {
    StepResult res;
    res.step = ++step_count;

    // --- traffic flow update (NPC) ---
    if (traffic_flow) {
        update_traffic_flow(dt);
    }

    const size_t n = cars.size();
    res.rewards.assign(n, 0.0f);
    res.done.assign(n, 0);
    res.status.assign(n, "ALIVE");
    res.agent_ids = agent_ids;

    // --- physics + base reward components (ego only) ---
    for (size_t i = 0; i < n; ++i) {
        if (!cars[i].alive) continue;
        const float thr = (i < throttles.size()) ? throttles[i] : 0.0f;
        const float st = (i < steerings.size()) ? steerings[i] : 0.0f;

        cars[i].update(thr, st, dt);
        cars[i].update_path_index();

        float r_prog = compute_progress(cars[i], reward_config);
        float r_stuck = compute_stuck(cars[i], reward_config);
        float r_smooth = compute_smooth(cars[i], reward_config);
        res.rewards[i] = r_prog + r_stuck + r_smooth;
    }

    // --- status per-agent (SUCCESS / CRASH_*) ---
    for (size_t i = 0; i < n; ++i) {
        if (!cars[i].alive) {
            res.done[i] = 1;
            res.status[i] = "DEAD";
            continue;
        }

        bool done = false;
        std::string status = "ALIVE";

        // SUCCESS（对齐 Python Scenario/agent.py::check_collision）
        if (cars[i].path.size() >= 2) {
            const auto end_pt = cars[i].path[cars[i].path.size() - 1];
            const auto prev_pt = cars[i].path[cars[i].path.size() - 2];
            const float dx_road = end_pt.first - prev_pt.first;
            const float dy_road = end_pt.second - prev_pt.second;

            constexpr float LATERAL_TOLERANCE = 15.0f;
            constexpr float LONGITUDINAL_TOLERANCE = 40.0f;

            bool is_success = false;
            if (std::fabs(dx_road) > std::fabs(dy_road)) {
                // Horizontal road
                const float lat_error = std::fabs(cars[i].state.y - end_pt.second);
                const float long_error = std::fabs(cars[i].state.x - end_pt.first);
                if (lat_error < LATERAL_TOLERANCE && long_error < LONGITUDINAL_TOLERANCE) {
                    is_success = true;
                }
            } else {
                // Vertical road
                const float lat_error = std::fabs(cars[i].state.x - end_pt.first);
                const float long_error = std::fabs(cars[i].state.y - end_pt.second);
                if (lat_error < LATERAL_TOLERANCE && long_error < LONGITUDINAL_TOLERANCE) {
                    is_success = true;
                }
            }

            if (is_success) {
                done = true;
                status = "SUCCESS";
            }
        }

        if (!done) {
            // 1) 屏幕边界：对齐 Python 的 out_of_screen 清理逻辑
            // Python ScenarioEnv._is_out_of_screen: margin=100
            // 注意：这里按“车身四角”判定，避免中心点仍在屏内但车身已出界的漏判。
            constexpr float MARGIN = 100.0f;

            bool out_of_screen = false;
            for (const auto& p : cars[i].corners()) {
                const float x = p.first;
                const float y = p.second;
                if (x < -MARGIN || x > float(WIDTH) + MARGIN || y < -MARGIN || y > float(HEIGHT) + MARGIN) {
                    out_of_screen = true;
                    break;
                }
            }

            if (out_of_screen) {
                done = true;
                status = "CRASH_WALL";
            } else {
                // 2) 撞墙/驶出路面：同样按“车身四角”判定更接近 Python 的 mask 碰撞效果
                bool off_road = false;
                auto on_road = [&](float x, float y) {
                    // Bitmap scenarios: use a small neighborhood check to reduce false CRASH_WALL
                    // due to int-cast quantization near lane boundaries (especially on slanted ramps).
                    const int xi = int(x);
                    const int yi = int(y);
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int xx = xi + dx;
                            const int yy = yi + dy;
                            if (xx < 0 || xx >= WIDTH || yy < 0 || yy >= HEIGHT) continue;
                            if (bitmap_road.at(xx, yy) > 0) return true;
                        }
                    }
                    return false;
                };
                
                for (const auto& p : cars[i].corners()) {
                    if (!on_road(p.first, p.second)) { off_road = true; break; }
                }

                if (off_road) {
                done = true;
                status = "CRASH_WALL";
                } else {
                // 3) 黄线压线：按车身四角 + 边缘采样，减少“中心点未压线但车身已压线”的漏判
                bool hit_line = false;
                auto hits_line = [&](float x, float y) {
                    if (use_bitmap_scenario) return bitmap_line.at(int(x), int(y)) > 0;
                    return geom.hits_yellow_line(x, y);
                };

                // 先检查四角
                for (const auto& p : cars[i].corners()) {
                    if (hits_line(p.first, p.second)) {
                        hit_line = true;
                        break;
                    }
                }
                // 再检查四条边的中点（更接近车身轮廓）
                if (!hit_line) {
                    const auto cs = cars[i].corners();
                    auto mid = [](const std::pair<float,float>& a, const std::pair<float,float>& b) {
                        return std::make_pair(0.5f * (a.first + b.first), 0.5f * (a.second + b.second));
                    };
                    const auto m0 = mid(cs[0], cs[1]);
                    const auto m1 = mid(cs[1], cs[2]);
                    const auto m2 = mid(cs[2], cs[3]);
                    const auto m3 = mid(cs[3], cs[0]);
                    
                    if (hits_line(m0.first, m0.second) ||
                        hits_line(m1.first, m1.second) ||
                        hits_line(m2.first, m2.second) ||
                        hits_line(m3.first, m3.second)) {
                        hit_line = true;
                    }
                }

                    if (hit_line) {
                        // CRASH_LINE is now a non-terminal penalty.
                        // We set the status so the reward calculation can pick it up,
                        // but we DO NOT set done = true.
                        status = "ON_LINE";
                    }
                }
            }
        }

        res.done[i] = done ? 1 : 0;
        res.status[i] = status;
    }

    // car-car collisions override (ego vs ego, ego vs npc)
    for (size_t i = 0; i < n; ++i) {
        if (!cars[i].alive || res.done[i]) continue;

        // vs other egos
        for (size_t j = i + 1; j < n; ++j) {
            if (!cars[j].alive || res.done[j]) continue;
            if (cars[i].check_collision(cars[j])) {
                res.done[i] = 1;
                res.done[j] = 1;
                res.status[i] = "CRASH_CAR";
                res.status[j] = "CRASH_CAR";
            }
        }

        // vs NPCs
        if (traffic_flow) {
            for (const auto& npc : traffic_cars) {
                if (!npc.alive) continue;
                if (cars[i].check_collision(npc)) {
                    res.done[i] = 1;
                    res.status[i] = "CRASH_CAR"; // Python treats ego-npc collision as CRASH_CAR
                    break; // One collision is enough
                }
            }
        }
    }

    // crash/success bonuses
    for (size_t i = 0; i < n; ++i) {
        // Step penalty for being on yellow line (non-terminal)
        if (res.status[i] == "ON_LINE") {
            res.rewards[i] += reward_config.k_cl;
        }

        if (!res.done[i]) continue;
        if (res.status[i] == "CRASH_CAR") res.rewards[i] += reward_config.k_cv;
        else if (res.status[i] == "CRASH_WALL") res.rewards[i] += reward_config.k_cw;
        else if (res.status[i] == "SUCCESS") res.rewards[i] += reward_config.k_succ;
    }

    // team reward mixing
    if (use_team_reward && n > 0) {
        float avg = 0.0f;
        for (float r : res.rewards) avg += r;
        avg /= float(n);
        for (size_t i = 0; i < n; ++i) {
            res.rewards[i] = (1.0f - reward_config.alpha) * res.rewards[i] + reward_config.alpha * avg;
        }
    }

    // --- respawn handling ---
    if (respawn_enabled) {
        for (size_t i = 0; i < n; ++i) {
            if (!cars[i].alive) continue;
            if (!res.done[i]) continue;
            if (res.status[i] == "CRASH_CAR" || res.status[i] == "CRASH_WALL") {
                cars[i].respawn();
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            if (res.done[i]) { res.terminated = true; break; }
        }
    }

    // terminated for respawn=True: only when all alive succeeded
    if (respawn_enabled) {
        int alive_cnt = 0;
        int succ_cnt = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!cars[i].alive) continue;
            alive_cnt++;
            if (res.done[i] && res.status[i] == "SUCCESS") succ_cnt++;
        }
        if (succ_cnt > 0 && succ_cnt == alive_cnt) res.terminated = true;
        res.agents_alive = alive_cnt;
    } else {
        int alive_cnt = 0;
        for (size_t i = 0; i < n; ++i) if (cars[i].alive) alive_cnt++;
        res.agents_alive = alive_cnt;
    }

    if (max_steps > 0 && step_count >= max_steps) res.truncated = true;

    // lidar update (after potential respawns, so next obs sees respawned state)
    // Avoid merging vectors: pass egos and NPCs as separate lists.
    static const std::vector<Car> kEmptyCars;

    const std::vector<Car>& cars2 = traffic_flow ? traffic_cars : kEmptyCars;

    for (size_t i = 0; i < n; ++i) {
        if (!cars[i].alive) continue;
        if (use_bitmap_scenario) {
            lidars[i].update_bitmap(cars[i], cars, cars2, bitmap_road, WIDTH, HEIGHT);
        } else {
            lidars[i].update(cars[i], cars, cars2, geom, WIDTH, HEIGHT);
        }
    }

    // Fill observation buffer and slice to res.obs to guarantee 145-dim output.
    update_observations_buffer();
    const int kObsDim = 145;
    res.obs.clear();
    res.obs.reserve(cars.size());
    for (size_t i = 0; i < cars.size(); ++i) {
        const float* row = obs_buffer.data() + i * (size_t)kObsDim;
        res.obs.emplace_back(row, row + kObsDim);
    }
    return res;
}

EnvState ScenarioEnv::get_state() const {
    EnvState s;
    s.cars.clear();
    s.cars.reserve(cars.size());
    for (const auto& c : cars) s.cars.push_back(c.get_dynamic_state());

    s.traffic_cars.clear();
    s.traffic_cars.reserve(traffic_cars.size());
    for (const auto& c : traffic_cars) s.traffic_cars.push_back(c.get_dynamic_state());

    s.agent_ids = agent_ids;
    s.next_agent_id = next_agent_id;
    s.step_count = step_count;
    return s;
}

void ScenarioEnv::set_state(const EnvState& s) {
    // Only restore dynamic state; paths and intentions are assumed constant within an episode.
    const size_t n = std::min(cars.size(), s.cars.size());
    for (size_t i = 0; i < n; ++i) {
        cars[i].set_dynamic_state(s.cars[i]);
    }

    const size_t tn = std::min(traffic_cars.size(), s.traffic_cars.size());
    for (size_t i = 0; i < tn; ++i) {
        traffic_cars[i].set_dynamic_state(s.traffic_cars[i]);
    }

    agent_ids = s.agent_ids;
    next_agent_id = s.next_agent_id;
    step_count = s.step_count;

    // Rebuild lidars to match car counts (counts are assumed unchanged within an episode)
    lidars.clear();
    lidars.resize(cars.size());
    traffic_lidars.clear();
    traffic_lidars.resize(traffic_cars.size());
}

std::vector<float> ScenarioEnv::get_global_state(int agent_index, int k_nearest) const {
    // Fixed-size CTDE state: ego + nearest-K egos (no NPCs).
    // Each vehicle state = [x_norm, y_norm, v_norm, heading_norm, intention, alive]
    // Output length = 6 * (1 + k_nearest)

    const int feat = 6;
    if (k_nearest < 0) k_nearest = 0;
    const int out_dim = feat * (1 + k_nearest);
    std::vector<float> out((size_t)out_dim, 0.0f);

    const int n = (int)cars.size();
    if (agent_index < 0 || agent_index >= n) return out;

    auto fill_state = [&](int slot, const Car& c) {
        const size_t base = (size_t)(slot * feat);
        if (!c.alive) {
            // keep zeros, but set alive flag explicitly
            out[base + 5] = 0.0f;
            return;
        }
        out[base + 0] = c.state.x / float(WIDTH);
        out[base + 1] = c.state.y / float(HEIGHT);
        out[base + 2] = c.state.v / PHYSICS_MAX_SPEED;
        out[base + 3] = c.state.heading / PI_F;
        out[base + 4] = float(c.intention);
        out[base + 5] = 1.0f;
    };

    // ego at slot 0
    fill_state(0, cars[(size_t)agent_index]);

    // compute distances to other egos
    struct Neighbor { float d2; int idx; };
    std::vector<Neighbor> neigh;
    neigh.reserve((size_t)std::max(0, n - 1));

    const float x0 = cars[(size_t)agent_index].state.x;
    const float y0 = cars[(size_t)agent_index].state.y;

    for (int j = 0; j < n; ++j) {
        if (j == agent_index) continue;
        const float dx = cars[(size_t)j].state.x - x0;
        const float dy = cars[(size_t)j].state.y - y0;
        neigh.push_back({dx * dx + dy * dy, j});
    }

    std::sort(neigh.begin(), neigh.end(), [](const Neighbor& a, const Neighbor& b) { return a.d2 < b.d2; });

    const int take = std::min(k_nearest, (int)neigh.size());
    for (int k = 0; k < take; ++k) {
        const int j = neigh[(size_t)k].idx;
        fill_state(1 + k, cars[(size_t)j]);
    }

    return out;
}

void ScenarioEnv::update_observations_buffer() {
    const int kObsDim = 145;
    const size_t n = cars.size();
    if (obs_buffer.size() != n * (size_t)kObsDim) {
        obs_buffer.assign(n * (size_t)kObsDim, 0.0f);
    }

    for (size_t i = 0; i < n; ++i) {
        float* obs = &obs_buffer[i * (size_t)kObsDim];
        std::fill(obs, obs + kObsDim, 0.0f);
        if (!cars[i].alive) continue;

        const float x = cars[i].state.x;
        const float y = cars[i].state.y;
        const float v = cars[i].state.v;
        const float heading = cars[i].state.heading;

        obs[0] = x / float(WIDTH);
        obs[1] = y / float(HEIGHT);
        obs[2] = v / PHYSICS_MAX_SPEED;
        obs[3] = heading / PI_F;

        float d_dst = 0.0f;
        float theta_error = 0.0f;
        if (!cars[i].path.empty()) {
            int lookahead = 10;
            int idx = cars[i].path_index;
            int target_idx = std::min(idx + lookahead, int(cars[i].path.size()) - 1);
            float tx = cars[i].path[target_idx].first;
            float ty = cars[i].path[target_idx].second;

            float dx_dest = tx - x;
            float dy_dest = ty - y;
            d_dst = std::sqrt(dx_dest * dx_dest + dy_dest * dy_dest) / float(WIDTH);

            float angle_to_target = std::atan2(-dy_dest, dx_dest);
            theta_error = wrap_angle_rad(angle_to_target - heading) / PI_F;
        }
        obs[4] = d_dst;
        obs[5] = theta_error;

        // --- Lane/Line features ---
        const auto corners = cars[i].corners();
        float off_road_count = 0.0f;
        float line_hit_count = 0.0f;

        for (const auto& p : corners) {
            if (!geom.is_on_road(p.first, p.second)) off_road_count += 1.0f;
            if (line_mask.is_line(int(p.first), int(p.second))) line_hit_count += 1.0f;
        }

        float cosH = std::cos(heading);
        float sinH = -std::sin(heading);
        float nx = -sinH;
        float ny = cosH;

        auto check_dist = [&](float dx, float dy, int max_steps) {
            for (int s = 1; s <= max_steps; ++s) {
                float cx = x + dx * s * 5.0f;
                float cy = y + dy * s * 5.0f;
                if (!geom.is_on_road(cx, cy)) return float(s * 5.0f) / 100.0f;
            }
            return 1.0f;
        };

        obs[6] = check_dist(nx, ny, 10);
        obs[7] = check_dist(-nx, -ny, 10);
        obs[8] = off_road_count / 4.0f;
        obs[9] = line_hit_count / 4.0f;

        float signed_cte = 0.0f;
        float path_heading_err = 0.0f;
        float in_lane = 0.0f;
        float lane_id_norm = 0.0f;

        if (!cars[i].path.empty()) {
            const int idx = std::max(0, std::min(cars[i].path_index, int(cars[i].path.size()) - 1));
            const int idx2 = std::min(idx + 5, int(cars[i].path.size()) - 1);
            const float px = cars[i].path[idx].first;
            const float py = cars[i].path[idx].second;
            const float px2 = cars[i].path[idx2].first;
            const float py2 = cars[i].path[idx2].second;
            const float tx = px2 - px;
            const float ty = py2 - py;
            const float tnorm = std::sqrt(tx * tx + ty * ty);
            if (tnorm > 1e-6f) {
                const float ux = tx / tnorm;
                const float uy = ty / tnorm;
                const float dxp = x - px;
                const float dyp = y - py;
                signed_cte = std::max(-1.0f, std::min(1.0f, (ux * dyp - uy * dxp) / 50.0f));
                path_heading_err = wrap_angle_rad(std::atan2(-uy, ux) - heading) / PI_F;
            }
        }

        if (use_bitmap_scenario) {
            const int li = bitmap_lane.at(int(x), int(y));
            if (li > 0) { in_lane = 1.0f; lane_id_norm = float(li) / 255.0f; }
        }

        obs[10] = signed_cte;
        obs[11] = path_heading_err;
        obs[12] = in_lane;
        obs[13] = lane_id_norm;

        struct NeighborRef { float d2; const Car* car; };
        std::vector<NeighborRef> neigh;
        neigh.reserve((cars.size() > 0 ? cars.size() - 1 : 0) + (traffic_flow ? traffic_cars.size() : 0));

        for (size_t j = 0; j < cars.size(); ++j) {
            if (j == i) continue;
            if (!cars[j].alive) continue;
            float dx = cars[j].state.x - x;
            float dy = cars[j].state.y - y;
            float d2 = dx * dx + dy * dy;
            neigh.push_back({d2, &cars[j]});
        }

        if (traffic_flow) {
            for (const auto& npc : traffic_cars) {
                if (!npc.alive) continue;
                float dx = npc.state.x - x;
                float dy = npc.state.y - y;
                float d2 = dx * dx + dy * dy;
                neigh.push_back({d2, &npc});
            }
        }

        const size_t take = std::min<size_t>(NEIGHBOR_COUNT, neigh.size());
        if (take > 0 && neigh.size() > take) {
            std::nth_element(neigh.begin(), neigh.begin() + take, neigh.end(),
                             [](const NeighborRef& a, const NeighborRef& b) { return a.d2 < b.d2; });
            neigh.resize(take);
        }
        std::sort(neigh.begin(), neigh.end(), [](const NeighborRef& a, const NeighborRef& b) { return a.d2 < b.d2; });

        size_t base = 14;
        for (size_t k = 0; k < take; ++k) {
            const Car* c = neigh[k].car;
            float dx = (c->state.x - x) / float(WIDTH);
            float dy = (c->state.y - y) / float(HEIGHT);
            float dv = (c->state.v - v) / PHYSICS_MAX_SPEED;
            float dtheta = wrap_angle_rad(c->state.heading - heading) / PI_F;
            float rel_x = c->state.x - x;
            float rel_y = c->state.y - y;

            obs[base + 0] = dx;
            obs[base + 1] = dy;
            obs[base + 2] = dv;
            obs[base + 3] = dtheta;
            obs[base + 4] = float(c->intention);
            obs[base + 5] = (rel_x * cosH - rel_y * sinH) / float(WIDTH);
            obs[base + 6] = (rel_x * sinH + rel_y * cosH) / float(WIDTH);
            base += 7;
        }

        const auto lidar_norm = lidars[i].normalized();
        const size_t lidar_base = 14 + 7 * NEIGHBOR_COUNT;
        for (size_t k = 0; k < lidar_norm.size() && (lidar_base + k) < (size_t)kObsDim; ++k) {
            obs[lidar_base + k] = lidar_norm[k];
        }
    }
}

std::vector<float> ScenarioEnv::get_observations_flat() const {
    const_cast<ScenarioEnv*>(this)->update_observations_buffer();
    return obs_buffer;
}

std::vector<std::vector<float>> ScenarioEnv::get_observations() const {
    std::vector<std::vector<float>> out;
    out.reserve(cars.size());

    for (size_t i = 0; i < cars.size(); ++i) {
        std::vector<float> obs;
        obs.assign(135, 0.0f); // New dimension: 127 + 8 = 135

        if (!cars[i].alive) {
            out.push_back(std::move(obs));
            continue;
        }

        const float x = cars[i].state.x;
        const float y = cars[i].state.y;
        const float v = cars[i].state.v;
        const float heading = cars[i].state.heading;

        obs[0] = x / float(WIDTH);
        obs[1] = y / float(HEIGHT);
        obs[2] = v / PHYSICS_MAX_SPEED;
        obs[3] = heading / PI_F;

        float d_dst = 0.0f;
        float theta_error = 0.0f;
        if (!cars[i].path.empty()) {
            int lookahead = 10;
            int idx = cars[i].path_index;
            int target_idx = std::min(idx + lookahead, int(cars[i].path.size()) - 1);
            float tx = cars[i].path[target_idx].first;
            float ty = cars[i].path[target_idx].second;

            float dx_dest = tx - x;
            float dy_dest = ty - y;
            d_dst = std::sqrt(dx_dest * dx_dest + dy_dest * dy_dest) / float(WIDTH);

            float angle_to_target = std::atan2(-dy_dest, dx_dest);
            theta_error = wrap_angle_rad(angle_to_target - heading) / PI_F;
        }
        obs[4] = d_dst;
        obs[5] = theta_error;

        // --- New Lane/Line explicit features (8 dims: 6-13) ---
        const auto corners = cars[i].corners();
        float off_road_count = 0.0f;
        float line_hit_count = 0.0f;
        float min_road_dist = 1.0f;
        float min_line_dist = 1.0f;

        for (const auto& p : corners) {
            if (!geom.is_on_road(p.first, p.second)) off_road_count += 1.0f;
            if (line_mask.is_line(int(p.first), int(p.second))) line_hit_count += 1.0f;
        }

        // Simple ray-cast sampling to approximate distance to boundaries (left/right)
        float cosH = std::cos(heading);
        float sinH = -std::sin(heading); 
        float nx = -sinH; // lateral normal
        float ny = cosH;

        auto check_dist = [&](float dx, float dy, int max_steps) {
            for (int s = 1; s <= max_steps; ++s) {
                float cx = x + dx * s * 5.0f;
                float cy = y + dy * s * 5.0f;
                if (!geom.is_on_road(cx, cy)) return float(s * 5.0f) / 100.0f;
            }
            return 1.0f;
        };

        obs[6] = check_dist(nx, ny, 10);  // dist to road edge left
        obs[7] = check_dist(-nx, -ny, 10); // dist to road edge right
        obs[8] = off_road_count / 4.0f;
        obs[9] = line_hit_count / 4.0f;
        
        // Use path-based lateral offset as an extra feature
        float lat_offset = 0.0f;
        if (!cars[i].path.empty()) {
            int idx = cars[i].path_index;
            float px = cars[i].path[idx].first;
            float py = cars[i].path[idx].second;
            lat_offset = std::sqrt((x-px)*(x-px) + (y-py)*(y-py)) / 50.0f;
        }
        obs[10] = std::min(1.0f, lat_offset);
        obs[11] = 0.0f; // reserved
        obs[12] = 0.0f; // reserved
        obs[13] = 0.0f; // reserved

        // Neighbor vehicles: include NPCs when traffic_flow is enabled
        struct NeighborRef {
            float dist;
            const Car* car;
        };

        std::vector<NeighborRef> neigh;
        neigh.reserve((cars.size() > 0 ? cars.size() - 1 : 0) + (traffic_flow ? traffic_cars.size() : 0));

        // other egos
        for (size_t j = 0; j < cars.size(); ++j) {
            if (j == i) continue;
            if (!cars[j].alive) continue;
            float dx = cars[j].state.x - x;
            float dy = cars[j].state.y - y;
            float dist = std::sqrt(dx * dx + dy * dy);
            neigh.push_back({dist, &cars[j]});
        }

        // NPCs
        if (traffic_flow) {
            for (const auto& npc : traffic_cars) {
                if (!npc.alive) continue;
                float dx = npc.state.x - x;
                float dy = npc.state.y - y;
                float dist = std::sqrt(dx * dx + dy * dy);
                neigh.push_back({dist, &npc});
            }
        }

        const size_t take = std::min<size_t>(NEIGHBOR_COUNT, neigh.size());
        if (take > 0 && neigh.size() > take) {
            std::nth_element(neigh.begin(), neigh.begin() + take, neigh.end(),
                             [](const NeighborRef& a, const NeighborRef& b) { return a.dist < b.dist; });
            neigh.resize(take);
        }
        std::sort(neigh.begin(), neigh.end(), [](const NeighborRef& a, const NeighborRef& b) { return a.dist < b.dist; });

        size_t base = 14; // Shifted from 6 to 14
        for (size_t k = 0; k < take; ++k) {
            const Car* c = neigh[k].car;
            float dx = (c->state.x - x) / float(WIDTH);
            float dy = (c->state.y - y) / float(HEIGHT);
            float dv = (c->state.v - v) / PHYSICS_MAX_SPEED;
            float dtheta = wrap_angle_rad(c->state.heading - heading) / PI_F;
            float intent = float(c->intention);

            obs[base + 0] = dx;
            obs[base + 1] = dy;
            obs[base + 2] = dv;
            obs[base + 3] = dtheta;
            obs[base + 4] = intent;
            base += 5;
        }

        const auto lidar_norm = lidars[i].normalized();
        const size_t lidar_base = 14 + 5 * NEIGHBOR_COUNT; // Shifted from 6 to 14
        for (size_t k = 0; k < lidar_norm.size() && (lidar_base + k) < obs.size(); ++k) {
            obs[lidar_base + k] = lidar_norm[k];
        }

        out.push_back(std::move(obs));
    }

    return out;
}
