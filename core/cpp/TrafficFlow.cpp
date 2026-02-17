#include "ScenarioEnv.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <iostream>

static constexpr float PI_F_TF = 3.14159265358979323846f;
static inline float wrap_angle_rad_tf(float a) {
    a = std::fmod(a + PI_F_TF, 2.0f * PI_F_TF);
    if (a < 0) a += 2.0f * PI_F_TF;
    return a - PI_F_TF;
}

// Helper: clamp index
static inline size_t clamp_idx(size_t i, size_t n) { return (n == 0) ? 0 : (i < n ? i : (n - 1)); }

static inline std::pair<float, float> plan_npc_action_tf(const Car& npc, const std::vector<const Car*>& /*all_vehicles*/) {
    float steer_cmd = 0.0f;
    if (npc.path.size() >= 2) {
        const float x = npc.state.x;
        const float y = npc.state.y;
        const float heading = npc.state.heading;
        const float v = std::max(0.1f, npc.state.v);

        const float Ld = std::max(40.0f, std::min(150.0f, 50.0f + 3.0f * v));

        int idx = std::max(0, npc.path_index);
        float acc_d = 0.0f;
        int target_idx = idx;
        for (int i = idx; i + 1 < (int)npc.path.size(); ++i) {
            acc_d += std::hypot(npc.path[i+1].first - npc.path[i].first, 
                                npc.path[i+1].second - npc.path[i].second);
            target_idx = i + 1;
            if (acc_d >= Ld) break;
        }

        const float tx = npc.path[target_idx].first;
        const float ty = npc.path[target_idx].second;

        const float dx = tx - x;
        const float dy = ty - y;
        const float angle_to_target = std::atan2(dy, dx);
        
        const float target_heading_math = -angle_to_target; 
        const float heading_err = wrap_angle_rad_tf(target_heading_math - heading);

        steer_cmd = std::max(-1.0f, std::min(1.0f, heading_err * 8.0f));
    }

    const float target_speed = PHYSICS_MAX_SPEED * 0.25f; 
    float acc_throttle = 0.0f;
    if (npc.state.v < target_speed) acc_throttle = 0.4f;
    else acc_throttle = -0.2f;

    return {acc_throttle, steer_cmd};
}

void ScenarioEnv::init_traffic_routes() {
    traffic_routes.clear();

    if (scenario_name.find("merge") != std::string::npos) {
        auto it_in_e = lane_layout.in_by_dir.find("E");
        auto it_out_e = lane_layout.out_by_dir.find("E");
        if (it_in_e != lane_layout.in_by_dir.end() && it_out_e != lane_layout.out_by_dir.end()) {
            const auto& in_lanes = it_in_e->second;
            const auto& out_lanes = it_out_e->second;
            for (size_t i = 0; i < in_lanes.size() && i < out_lanes.size(); ++i) {
                traffic_routes.emplace_back(in_lanes[i], out_lanes[i]);
            }
        }

        if (lane_layout.points.find("IN_RAMP_1") != lane_layout.points.end()) {
            for (int k = 0; k < 10; ++k) {
                traffic_routes.emplace_back("IN_RAMP_1", "OUT_2");
            }
        }
        return;
    }

    if (scenario_name.find("highway") != std::string::npos) {
        for (const auto& direction : lane_layout.dir_order) {
            auto it_in = lane_layout.in_by_dir.find(direction);
            auto it_out = lane_layout.out_by_dir.find(direction);
            if (it_in == lane_layout.in_by_dir.end() || it_out == lane_layout.out_by_dir.end()) continue;

            const auto& in_lanes = it_in->second;
            const auto& out_lanes = it_out->second;

            for (size_t i = 0; i < in_lanes.size(); ++i) {
                if (i < out_lanes.size()) {
                    traffic_routes.emplace_back(in_lanes[i], out_lanes[i]);
                }
            }
        }
        return;
    }

    const auto &dir_order = lane_layout.dir_order;
    const std::unordered_map<std::string, std::string> opposite = {
        {"N", "S"}, {"S", "N"}, {"E", "W"}, {"W", "E"}
    };
    const std::unordered_map<std::string, std::string> left_turn = {
        {"N", "E"}, {"E", "S"}, {"S", "W"}, {"W", "N"}
    };

    for (const auto &direction : dir_order) {
        auto it_in = lane_layout.in_by_dir.find(direction);
        if (it_in == lane_layout.in_by_dir.end()) continue;

        const auto &in_lanes = it_in->second;
        auto it_straight_out = lane_layout.out_by_dir.find(opposite.at(direction));
        auto it_left_out = lane_layout.out_by_dir.find(left_turn.at(direction));
        const auto &straight_out_lanes = (it_straight_out != lane_layout.out_by_dir.end()) ? it_straight_out->second : std::vector<std::string>{};
        const auto &left_out_lanes = (it_left_out != lane_layout.out_by_dir.end()) ? it_left_out->second : std::vector<std::string>{};

        for (const auto &start_id : in_lanes) {
            size_t idx = 0;
            auto it_idx = lane_layout.idx_of.find(start_id);
            if (it_idx != lane_layout.idx_of.end()) idx = size_t(std::max(0, it_idx->second));

            if (!straight_out_lanes.empty()) {
                const auto &out_id = straight_out_lanes[clamp_idx(idx, straight_out_lanes.size())];
                traffic_routes.emplace_back(start_id, out_id);
            }
            if (!left_out_lanes.empty()) {
                const auto &out_id = left_out_lanes[clamp_idx(idx, left_out_lanes.size())];
                traffic_routes.emplace_back(start_id, out_id);
            }
        }
    }
}

bool ScenarioEnv::is_spawn_blocked(float sx, float sy) const {
    Car probe;
    probe.state.x = sx;
    probe.state.y = sy;
    probe.state.v = 0.0f;
    probe.state.heading = 0.0f;

    for (const auto &c : cars) if (probe.check_collision(c)) return true;
    for (const auto &c : traffic_cars) if (probe.check_collision(c)) return true;
    return false;
}

bool ScenarioEnv::is_arrived(const Car &car, float tol) const {
    if (car.path.empty()) return false;
    const auto goal = car.path.back();
    float d = std::hypot(car.state.x - goal.first, car.state.y - goal.second);
    return d < tol;
}

bool ScenarioEnv::is_out_of_screen(const Car &car, float margin) const {
    const float x = car.state.x;
    const float y = car.state.y;
    if (x < -margin || x > float(WIDTH) + margin || y < -margin || y > float(HEIGHT) + margin) return true;
    return false;
}

static inline void teleport_offscreen(Car& npc) {
    npc.alive = false;
    npc.state.x = -1e6f;
    npc.state.y = -1e6f;
    npc.state.v = 0.0f;
    npc.acc = 0.0f;
    npc.steering_angle = 0.0f;
    npc.path_index = 0;
}

static inline int compute_target_npc_count(float density, int kmax) {
    if (kmax <= 0) return 0;
    float d = std::max(0.0f, std::min(1.0f, density));
    int k = int(std::lround(d * float(kmax)));
    return std::max(0, std::min(k, kmax));
}

static bool spawn_traffic_car_into_slot(ScenarioEnv& env, size_t slot) {
    if (env.traffic_routes.empty()) return false;

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(0, env.traffic_routes.size() - 1);

    const auto &route = env.traffic_routes[dist(rng)];
    const auto it = env.lane_layout.points.find(route.first);
    if (it == env.lane_layout.points.end()) return false;

    float sx = it->second.first;
    float sy = it->second.second;

    int intent = INTENT_STRAIGHT;
    auto it_int = env.route_intents.find({route.first, route.second});
    if (it_int != env.route_intents.end()) intent = it_int->second;
    else intent = determine_intent(env.lane_layout, route.first, route.second);

    std::vector<std::pair<float,float>> path;
    if (env.scenario_name.find("roundabout") != std::string::npos) {
        path = generate_path_roundabout_cpp(env.lane_layout, env.num_lanes, intent, route.first, route.second);
    } else if (env.scenario_name.find("bottleneck") != std::string::npos) {
        path = generate_path_bottleneck_cpp(env.lane_layout, env.num_lanes, route.first, route.second);
    } else {
        path = generate_path_cpp(env.lane_layout, env.num_lanes, intent, route.first, route.second);
    }
    if (path.size() < 2) return false;

    auto is_blocked_with_heading = [&](float x, float y) {
        float dx = path[1].first - path[0].first;
        float dy = path[1].second - path[0].second;
        float h = std::atan2(-dy, dx);

        Car probe;
        probe.state.x = x;
        probe.state.y = y;
        probe.state.v = 0.0f;
        probe.state.heading = h;

        for (const auto &c : env.cars) if (probe.check_collision(c)) return true;
        for (const auto &c : env.traffic_cars) if (probe.check_collision(c)) return true;
        return false;
    };

    if (is_blocked_with_heading(sx, sy)) {
        float dx = path[1].first - path[0].first;
        float dy = path[1].second - path[0].second;
        float len = std::hypot(dx, dy);
        if (len <= 1e-6f) return false;

        float ax = sx + (dx / len) * (2.0f * CAR_LENGTH);
        float ay = sy + (dy / len) * (2.0f * CAR_LENGTH);
        if (is_blocked_with_heading(ax, ay)) return false;

        sx = ax;
        sy = ay;
    }

    float heading = std::atan2(-(path[1].second - path[0].second), path[1].first - path[0].first);

    Car npc;
    npc.state.x = sx;
    npc.state.y = sy;
    npc.state.v = 0.0f;
    npc.state.heading = heading;
    npc.spawn_state = npc.state;
    npc.alive = true;
    npc.intention = intent;
    npc.path = std::move(path);
    npc.path_index = 0;
    npc.prev_dist_to_goal = 0.0f;
    npc.prev_action = {0.0f, 0.0f};

    if (slot >= env.traffic_cars.size()) {
        env.traffic_cars.resize(slot + 1);
        env.traffic_lidars.resize(slot + 1);
    }
    env.traffic_cars[slot] = std::move(npc);
    return true;
}

void ScenarioEnv::try_spawn_traffic_car() {
    if (traffic_routes.empty()) return;
    size_t new_slot = traffic_cars.size();
    spawn_traffic_car_into_slot(*this, new_slot);
}

void ScenarioEnv::update_traffic_flow(float dt) {
    if (!traffic_flow) return;

    if (traffic_mode == TrafficMode::STOCHASTIC) {
        const float arrival_rate = traffic_density;
        const float spawn_prob = 1.0f - std::exp(-arrival_rate * dt);
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);

        if (uni(rng) < spawn_prob) {
            try_spawn_traffic_car();
        }

        std::vector<const Car*> all_vehicles;
        all_vehicles.reserve(traffic_cars.size());
        for (const auto& c : traffic_cars) all_vehicles.push_back(&c);

        for (auto& npc : traffic_cars) {
            if (!npc.alive) continue;
            npc.update_path_index();
            const auto action = plan_npc_action_tf(npc, all_vehicles);
            npc.update(action.first, action.second, dt);
            npc.update_path_index();
        }

        for (size_t i = 0; i < traffic_cars.size(); ++i) {
            if (!traffic_cars[i].alive) continue;
            for (size_t j = i + 1; j < traffic_cars.size(); ++j) {
                if (!traffic_cars[j].alive) continue;
                if (traffic_cars[i].check_collision(traffic_cars[j])) {
                    traffic_cars[i].alive = false;
                    traffic_cars[j].alive = false;
                }
            }
        }

        for (size_t i = 0; i < traffic_cars.size();) {
            if (!traffic_cars[i].alive || is_arrived(traffic_cars[i], 20.0f) || is_out_of_screen(traffic_cars[i], 100.0f)) {
                traffic_cars.erase(traffic_cars.begin() + long(i));
                traffic_lidars.erase(traffic_lidars.begin() + long(i));
                continue;
            }
            ++i;
        }
        return;
    }

    const int K = compute_target_npc_count(traffic_density, traffic_kmax);
    if (K <= 0) {
        traffic_cars.clear();
        traffic_lidars.clear();
        return;
    }

    if ((int)traffic_cars.size() != K) {
        traffic_cars.resize((size_t)K);
        traffic_lidars.resize((size_t)K);
        for (int i = 0; i < K; ++i) teleport_offscreen(traffic_cars[(size_t)i]);
    }

    if (!traffic_freeze) {
        for (int i = 0; i < K; ++i) {
            if (!traffic_cars[(size_t)i].alive) spawn_traffic_car_into_slot(*this, (size_t)i);
        }
    }

    std::vector<const Car*> all_vehicles;
    all_vehicles.reserve(traffic_cars.size());
    for (const auto& c : traffic_cars) if (c.alive) all_vehicles.push_back(&c);

    for (auto& npc : traffic_cars) {
        if (!npc.alive) continue;
        npc.update_path_index();
        const auto action = plan_npc_action_tf(npc, all_vehicles);
        npc.update(action.first, action.second, dt);
        npc.update_path_index();
        if (is_arrived(npc, 20.0f) || is_out_of_screen(npc, 100.0f)) teleport_offscreen(npc);
    }

    for (size_t i = 0; i < traffic_cars.size(); ++i) {
        if (!traffic_cars[i].alive) continue;
        for (size_t j = i + 1; j < traffic_cars.size(); ++j) {
            if (!traffic_cars[j].alive) continue;
            if (traffic_cars[i].check_collision(traffic_cars[j])) {
                teleport_offscreen(traffic_cars[i]);
                teleport_offscreen(traffic_cars[j]);
            }
        }
    }
}
