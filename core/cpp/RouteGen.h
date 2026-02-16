#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

struct LaneLayout {
    std::unordered_map<std::string, std::pair<float, float>> points;
    std::unordered_map<std::string, std::vector<std::string>> in_by_dir;
    std::unordered_map<std::string, std::vector<std::string>> out_by_dir;
    std::unordered_map<std::string, std::string> dir_of; // N/E/S/W
    std::unordered_map<std::string, int> idx_of;
    std::vector<std::string> dir_order;
};

enum Intent {
    INTENT_STRAIGHT = 0,
    INTENT_LEFT = 1,
    INTENT_RIGHT = 2,
};

// Build layout points for given num_lanes (mirrors Scenario.agent.build_lane_layout)
LaneLayout build_lane_layout_cpp(int num_lanes);
LaneLayout build_lane_layout_t_junction_cpp(int num_lanes);
LaneLayout build_lane_layout_highway_cpp(int num_lanes);
LaneLayout build_lane_layout_merge_cpp(int num_lanes);
LaneLayout build_lane_layout_bottleneck_cpp(int num_lanes);
LaneLayout build_lane_layout_roundabout_cpp(int num_lanes);

// Determine intent from start/end lane ids
int determine_intent(const LaneLayout& layout, const std::string& start_id, const std::string& end_id);

// Generate path points (mirrors Scenario.agent.Car._generate_path)
std::vector<std::pair<float,float>> generate_path_cpp(const LaneLayout& layout,
                                                      int num_lanes,
                                                      int intent,
                                                      const std::string& start_id,
                                                      const std::string& end_id);

std::vector<std::pair<float,float>> generate_path_bottleneck_cpp(const LaneLayout& layout,
                                                                 int num_lanes,
                                                                 const std::string& start_id,
                                                                 const std::string& end_id);

std::vector<std::pair<float,float>> generate_path_roundabout_cpp(const LaneLayout& layout,
                                                                 int num_lanes,
                                                                 int intent,
                                                                 const std::string& start_id,
                                                                 const std::string& end_id);
