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

// --- C++ port (incremental) of Scenario/agent.py::Car.plan_autonomous_action ---
// 当前版本实现：
// - 横向：路径 lookahead 的 heading_error * 3.0
// - 纵向：巡航到 target_speed=PHYSICS_MAX_SPEED*0.6，并做前车跟车制动
// - 鬼影路径扫描：检测Scenario冲突并分级制动（简化版）
static inline std::pair<float, float> plan_npc_action_tf(const Car& npc, const std::vector<const Car*>& /*all_vehicles*/) {
    // --- 1) 严格横向控制：Pure Pursuit 增强版 ---
    float steer_cmd = 0.0f;
    if (npc.path.size() >= 2) {
        const float x = npc.state.x;
        const float y = npc.state.y;
        const float heading = npc.state.heading;
        const float v = std::max(0.1f, npc.state.v);

        // 增加前视距离以提前应对 90 度弯道
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

        // 统一使用屏幕坐标系计算误差 (y 轴向下)
        const float dx = tx - x;
        const float dy = ty - y;
        const float angle_to_target = std::atan2(dy, dx);
        
        // 关键点：Car::update 里的 position 更新是 y -= v*sin(h)，这对应数学坐标系
        // 我们需要把屏幕坐标系的 angle_to_target 转换回车辆的数学坐标系
        const float target_heading_math = -angle_to_target; 
        const float heading_err = wrap_angle_rad_tf(target_heading_math - heading);

        // 增加增益，使转向更灵敏
        steer_cmd = std::max(-1.0f, std::min(1.0f, heading_err * 8.0f));
    }

    // --- 2) 纵向控制：更稳定的巡航 ---
    const float target_speed = PHYSICS_MAX_SPEED * 0.25f; 
    float acc_throttle = 0.0f;
    if (npc.state.v < target_speed) acc_throttle = 0.4f;
    else acc_throttle = -0.2f;

    return {acc_throttle, steer_cmd};
}

void ScenarioEnv::init_traffic_routes() {
    traffic_routes.clear();

    // Merge scenario: allow traffic to also spawn from ramp.
    // We rely on lane_layout providing an entry point "IN_RAMP_1" and RouteGen providing a valid path.
    if (scenario_name.find("merge") != std::string::npos) {
        // Main road lanes (if any)
        auto it_in_e = lane_layout.in_by_dir.find("E");
        auto it_out_e = lane_layout.out_by_dir.find("E");
        if (it_in_e != lane_layout.in_by_dir.end() && it_out_e != lane_layout.out_by_dir.end()) {
            const auto& in_lanes = it_in_e->second;
            const auto& out_lanes = it_out_e->second;
            for (size_t i = 0; i < in_lanes.size() && i < out_lanes.size(); ++i) {
                traffic_routes.emplace_back(in_lanes[i], out_lanes[i]);
            }
        }

        // Ramp -> lane 2 exit (weight = 10 to make it much more frequent)
        if (lane_layout.points.find("IN_RAMP_1") != lane_layout.points.end()) {
            for (int k = 0; k < 10; ++k) {
                traffic_routes.emplace_back("IN_RAMP_1", "OUT_2");
            }
        }
        
        std::cerr << "[TrafficFlow] Init Merge: " << traffic_routes.size() 
                  << " routes. Ramp (IN_RAMP_1 -> OUT_2) included." << std::endl;
        return;
    }

    if (scenario_name.find("highway") != std::string::npos) {
        // Highway: One-way straight-through routes (IN_i -> OUT_i)
        for (const auto& direction : lane_layout.dir_order) {
            auto it_in = lane_layout.in_by_dir.find(direction);
            auto it_out = lane_layout.out_by_dir.find(direction);
            if (it_in == lane_layout.in_by_dir.end() || it_out == lane_layout.out_by_dir.end()) continue;

            const auto& in_lanes = it_in->second;
            const auto& out_lanes = it_out->second;

            // Log available lanes for debugging
            std::cout << "[TrafficFlow] Init highway routes for dir=" << direction 
                      << " in_size=" << in_lanes.size() << " out_size=" << out_lanes.size() << std::endl;

            for (size_t i = 0; i < in_lanes.size(); ++i) {
                if (i < out_lanes.size()) {
                    traffic_routes.emplace_back(in_lanes[i], out_lanes[i]);
                    std::cout << "[TrafficFlow]   Added route: " << in_lanes[i] << " -> " << out_lanes[i] << std::endl;
                }
            }
        }
        return;
    }

    // Mirror Scenario/env.py::_init_traffic_routes fallback (straight + left using lane indices)
    // dir_order in python: ['N','E','S','W']
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
    // 构造一个临时的探测车，用于精确碰撞检测
    Car probe;
    probe.state.x = sx;
    probe.state.y = sy;
    probe.state.v = 0.0f;
    probe.state.heading = 0.0f; // 初始探测朝向，后面在 try_spawn 里会根据路径校准

    // 检查所有主车
    for (const auto &c : cars) {
        if (probe.check_collision(c)) return true;
    }

    // 检查所有 NPC 车
    for (const auto &c : traffic_cars) {
        if (probe.check_collision(c)) return true;
    }

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

void ScenarioEnv::try_spawn_traffic_car() {
    if (traffic_routes.empty()) return;

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(0, traffic_routes.size() - 1);

    const auto &route = traffic_routes[dist(rng)];
    const auto it = lane_layout.points.find(route.first);
    if (it == lane_layout.points.end()) return;

    float sx = it->second.first;
    float sy = it->second.second;

    // --- Generate path first to know the heading (also needed for forward offset direction) ---
    int intent = INTENT_STRAIGHT;
    auto it_int = route_intents.find({route.first, route.second});
    if (it_int != route_intents.end()) intent = it_int->second;
    else intent = determine_intent(lane_layout, route.first, route.second);

    std::vector<std::pair<float,float>> path;
    if (scenario_name.find("roundabout") != std::string::npos) {
        path = generate_path_roundabout_cpp(lane_layout, num_lanes, intent, route.first, route.second);
    } else if (scenario_name.find("bottleneck") != std::string::npos) {
        path = generate_path_bottleneck_cpp(lane_layout, num_lanes, route.first, route.second);
    } else {
        path = generate_path_cpp(lane_layout, num_lanes, intent, route.first, route.second);
    }
    if (path.size() < 2) return;

    auto is_blocked_with_heading = [&](float x, float y) {
        float dx = path[1].first - path[0].first;
        float dy = path[1].second - path[0].second;
        float h = std::atan2(-dy, dx);

        Car probe;
        probe.state.x = x;
        probe.state.y = y;
        probe.state.v = 0.0f;
        probe.state.heading = h;

        for (const auto &c : cars) if (probe.check_collision(c)) return true;
        for (const auto &c : traffic_cars) if (probe.check_collision(c)) return true;
        return false;
    };

    if (is_blocked_with_heading(sx, sy)) {
        float dx = path[1].first - path[0].first;
        float dy = path[1].second - path[0].second;
        float len = std::hypot(dx, dy);
        if (len <= 1e-6f) return;

        float ax = sx + (dx / len) * (2.0f * CAR_LENGTH);
        float ay = sy + (dy / len) * (2.0f * CAR_LENGTH);
        if (is_blocked_with_heading(ax, ay)) return;

        sx = ax;
        sy = ay;
    }

    float heading = 0.0f;
    {
        float dx = path[1].first - path[0].first;
        float dy = path[1].second - path[0].second;
        heading = std::atan2(-dy, dx);
    }

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

    traffic_cars.push_back(std::move(npc));
    traffic_lidars.emplace_back();
}

void ScenarioEnv::update_traffic_flow(float dt) {
    if (!traffic_flow) return;

    // Spawn probability: 1 - exp(-arrival_rate * dt)
    const float arrival_rate = traffic_density;
    const float spawn_prob = 1.0f - std::exp(-arrival_rate * dt);

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    if (uni(rng) < spawn_prob) {
        try_spawn_traffic_car();
    }

    // --- NPC Controller Update ---
    // NPC planning ignores ego vehicles: only consider NPC traffic cars
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

    // --- NPC-NPC collision: remove both (match Scenario/env.py behavior) ---
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

    // Remove arrived / out-of-screen / collided
    for (size_t i = 0; i < traffic_cars.size();) {
        if (!traffic_cars[i].alive || is_arrived(traffic_cars[i], 20.0f) || is_out_of_screen(traffic_cars[i], 100.0f)) {
            traffic_cars.erase(traffic_cars.begin() + long(i));
            traffic_lidars.erase(traffic_lidars.begin() + long(i));
            continue;
        }
        ++i;
    }
}
