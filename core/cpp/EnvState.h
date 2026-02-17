#pragma once
#include <vector>
#include "Car.h"

// Snapshot state for fast MCTS rollbacks.
// Optimized: Only stores dynamic state to avoid expensive path vector copies.
struct EnvState {
    std::vector<CarDynamicState> cars;
    std::vector<CarDynamicState> traffic_cars;
    std::vector<long long> agent_ids;
    long long next_agent_id{1};
    int step_count{0};
};
