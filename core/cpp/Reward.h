#pragma once
#include <string>
#include <vector>

struct RewardConfig {
    float k_prog{10.0f};
    float v_min_ms{1.0f};
    float k_stuck{-0.01f};
    float k_cv{-10.0f};
    float k_cw{-5.0f};    // Crash Wall/Off-road
    float k_cl{-2.0f};    // Crash Line (Yellow line)
    float k_succ{10.0f};
    float k_sm{-0.02f};
    float alpha{0.2f};
};

struct StepResult {
    std::vector<std::vector<float>> obs; // (N,135)
    std::vector<float> rewards;          // (N)
    std::vector<int> done;               // (N)
    std::vector<std::string> status;     // (N)

    // Info parity with Scenario/env.py
    std::vector<long long> agent_ids;    // stable per-agent ids (C++ side)
    int agents_alive{0};

    bool terminated{false};
    bool truncated{false};
    int step{0};
};
