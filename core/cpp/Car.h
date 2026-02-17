#pragma once
#include <array>
#include <cmath>
#include <vector>
#include <utility>

#include "constants.h"

struct State {
    float x{0.0f};
    float y{0.0f};
    float v{0.0f};       // px/frame (matches Scenario.agent.Car.speed)
    float heading{0.0f}; // radians
};

// Lightweight dynamic state for snapshots/MCTS
struct CarDynamicState {
    State state;
    float acc{0.0f};
    float steering_angle{0.0f};
    bool alive{true};
    int path_index{0};
    float prev_dist_to_goal{0.0f};
    std::pair<float, float> prev_action{0.0f, 0.0f};
};

class Car {
public:
    State state;
    float length{54.0f};
    float width{24.0f};

    // Control state (mirrors Scenario.agent.Car)
    float acc{0.0f};           // px/frame^2 equivalent (acc*DT updates speed)
    float steering_angle{0.0f};

    // Life-cycle
    bool alive{true};
    State spawn_state;

    // Navigation & Reward state
    int intention{0};
    std::vector<std::pair<float, float>> path;
    int path_index{0};

    float prev_dist_to_goal{0.0f};
    std::pair<float, float> prev_action{0.0f, 0.0f}; // [acc/MAX_ACC, steering/MAX_STEERING_ANGLE]

    void update(float throttle, float steer_input, float dt);
    bool check_collision(const Car &other) const;
    std::array<std::pair<float,float>,4> corners() const;

    void set_path(std::vector<std::pair<float,float>> p);
    void update_path_index();

    void respawn();

    // Snapshot helpers
    CarDynamicState get_dynamic_state() const {
        return {state, acc, steering_angle, alive, path_index, prev_dist_to_goal, prev_action};
    }
    void set_dynamic_state(const CarDynamicState& ds) {
        state = ds.state;
        acc = ds.acc;
        steering_angle = ds.steering_angle;
        alive = ds.alive;
        path_index = ds.path_index;
        prev_dist_to_goal = ds.prev_dist_to_goal;
        prev_action = ds.prev_action;
    }
};
