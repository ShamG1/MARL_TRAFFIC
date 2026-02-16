#pragma once
#include <vector>
#include "Car.h"

// Snapshot state for fast MCTS rollbacks.
// 说明：这里只保存可恢复环境动力学/观测所需的最小状态。
// - ego cars + traffic cars 的完整 Car 对象（包含 state/path/intention/path_index 等）
// - step_count / next_agent_id
struct EnvState {
    std::vector<Car> cars;
    std::vector<Car> traffic_cars;
    std::vector<long long> agent_ids;
    long long next_agent_id{1};
    int step_count{0};
};
