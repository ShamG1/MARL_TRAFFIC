#include "RouteGen.h"
#include "constants.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

static constexpr float PI_F = 3.14159265358979323846f;
static constexpr float FILLET_RADIUS_BASE = 126.0f; 

// ==========================================
// 辅助函数
// ==========================================
static std::pair<float,float> get_stop_line_point(const std::pair<float,float>& pt, const std::string& dir, int num_lanes) {
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;
    float rw = num_lanes * LANE_WIDTH_PX;
    float off = rw + 30.0f; 

    if (dir == "N") return std::make_pair(pt.first, CY - off);
    if (dir == "S") return std::make_pair(pt.first, CY + off);
    if (dir == "E") return std::make_pair(CX + off, pt.second);
    if (dir == "W") return std::make_pair(CX - off, pt.second);
    return pt;
}

static std::pair<float,float> get_lane_dash_end_point(const std::pair<float,float>& pt, const std::string& dir, int num_lanes) {
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;
    float rw = num_lanes * LANE_WIDTH_PX;
    float stop_off = rw + CORNER_RADIUS;

    if (dir == "N") return std::make_pair(pt.first, CY - stop_off);
    if (dir == "S") return std::make_pair(pt.first, CY + stop_off);
    if (dir == "E") return std::make_pair(CX + stop_off, pt.second);
    if (dir == "W") return std::make_pair(CX - stop_off, pt.second);
    return pt;
}

static float normalize_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a <= -PI_F) a += 2.0f * PI_F;
    return a;
}

// ==========================================
// 基础布局 (保持你原有的逻辑，不做修改)
// ==========================================
LaneLayout build_lane_layout_cpp(int num_lanes) {
    LaneLayout layout;
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;
    const float MARGIN = 30.0f;

    const char dir_order_arr[4] = {'N','E','S','W'};
    layout.dir_order = {"N", "E", "S", "W"};
    layout.in_by_dir = {{"N", {}}, {"E", {}}, {"S", {}}, {"W", {}}};
    layout.out_by_dir = {{"N", {}}, {"E", {}}, {"S", {}}, {"W", {}}};

    for (int d_idx = 0; d_idx < 4; ++d_idx) {
        char d = dir_order_arr[d_idx];
        for (int j = 0; j < num_lanes; ++j) {
            float offset = LANE_WIDTH_PX * (0.5f + float(j));
            std::string in_name = "IN_" + std::to_string(d_idx * num_lanes + j + 1);
            std::string out_name = "OUT_" + std::to_string(d_idx * num_lanes + j + 1);

            // 保持你原有的坐标计算逻辑
            float in_x=0, in_y=0, out_x=0, out_y=0;
            if (d=='N') {
                in_x = CX - offset; in_y = MARGIN;
                out_x = CX + offset; out_y = MARGIN;
            } else if (d=='S') {
                in_x = CX + offset; in_y = HEIGHT - MARGIN;
                out_x = CX - offset; out_y = HEIGHT - MARGIN;
            } else if (d=='E') {
                in_x = WIDTH - MARGIN; in_y = CY - offset;
                out_x = WIDTH - MARGIN; out_y = CY + offset;
            } else { // W
                in_x = MARGIN; in_y = CY + offset;
                out_x = MARGIN; out_y = CY - offset;
            }

            layout.points[in_name] = {in_x, in_y};
            layout.points[out_name] = {out_x, out_y};
            layout.dir_of[in_name] = std::string(1, d);
            layout.dir_of[out_name] = std::string(1, d);
            layout.idx_of[in_name] = j;
            layout.idx_of[out_name] = j;

            layout.in_by_dir[std::string(1, d)].push_back(in_name);
            layout.out_by_dir[std::string(1, d)].push_back(out_name);
        }
    }
    return layout;
}

LaneLayout build_lane_layout_t_junction_cpp(int num_lanes) {
    LaneLayout layout;
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;
    const float MARGIN = 30.0f;

    // 重新定义的 3 向顺序：East, South, West
    const std::vector<std::string> t_dir_order = {"E", "S", "W"};
    layout.dir_order = {"E", "S", "W"};
    layout.in_by_dir = {{"E", {}}, {"S", {}}, {"W", {}}};
    layout.out_by_dir = {{"E", {}}, {"S", {}}, {"W", {}}};

    for (int d_idx = 0; d_idx < (int)t_dir_order.size(); ++d_idx) {
        std::string d = t_dir_order[d_idx];
        for (int j = 0; j < num_lanes; ++j) {
            float offset = LANE_WIDTH_PX * (0.5f + float(j));
            // 编号从 1 开始连续递增
            int global_idx = d_idx * num_lanes + j + 1;
            std::string in_name = "IN_" + std::to_string(global_idx);
            std::string out_name = "OUT_" + std::to_string(global_idx);

            float in_x=0, in_y=0, out_x=0, out_y=0;
            if (d=="S") {
                in_x = CX + offset; in_y = HEIGHT - MARGIN;
                out_x = CX - offset; out_y = HEIGHT - MARGIN;
            } else if (d=="E") {
                in_x = WIDTH - MARGIN; in_y = CY - offset;
                out_x = WIDTH - MARGIN; out_y = CY + offset;
            } else if (d=="W") {
                in_x = MARGIN; in_y = CY + offset;
                out_x = MARGIN; out_y = CY - offset;
            }

            layout.points[in_name] = {in_x, in_y};
            layout.points[out_name] = {out_x, out_y};
            layout.dir_of[in_name] = d;
            layout.dir_of[out_name] = d;
            layout.idx_of[in_name] = j;
            layout.idx_of[out_name] = j;

            layout.in_by_dir[d].push_back(in_name);
            layout.out_by_dir[d].push_back(out_name);
        }
    }
    return layout;
}

LaneLayout build_lane_layout_highway_cpp(int num_lanes) {
    LaneLayout layout;
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;
    const float MARGIN = 30.0f;

    // One-way highway: East-bound only (W -> E)
    const std::vector<std::string> highway_dirs = {"E"};
    layout.dir_order = highway_dirs;
    layout.in_by_dir = {{"E", {}}};
    layout.out_by_dir = {{"E", {}}};

    layout.in_by_dir["E"] = {};
    layout.out_by_dir["E"] = {};

    for (int j = 0; j < num_lanes; ++j) {
        float total_rw = num_lanes * LANE_WIDTH_PX;
        float top_edge = CY - (total_rw / 2.0f);
        float offset_in_road = (j + 0.5f) * LANE_WIDTH_PX;
        float y_pos = top_edge + offset_in_road;

        // IDs from 1 to N
        int global_idx = j + 1;
        std::string in_name = "IN_" + std::to_string(global_idx);
        std::string out_name = "OUT_" + std::to_string(global_idx);

        // Spawn on West side (left), exit on East side (right)
        float in_x = MARGIN;
        float out_x = WIDTH - MARGIN;

        layout.points[in_name] = {in_x, y_pos};
        layout.points[out_name] = {out_x, y_pos};
        layout.dir_of[in_name] = "E";
        layout.dir_of[out_name] = "E";
        layout.idx_of[in_name] = j;
        layout.idx_of[out_name] = j;

        layout.in_by_dir["E"].push_back(in_name);
        layout.out_by_dir["E"].push_back(out_name);
    }
    return layout;
}

LaneLayout build_lane_layout_merge_cpp(int num_lanes) {
    LaneLayout layout;
    const float CY = HEIGHT * 0.5f;
    const float MARGIN = 30.0f;
    const float LW = LANE_WIDTH_PX;

    // Merge: Main road (W->E) + Ramp (SW->Join)
    layout.dir_order = {"E", "R"};
    layout.in_by_dir = {{"E", {}}, {"R", {}}};
    layout.out_by_dir = {{"E", {}}};

    // Main Road (fixed 2 lanes for this specific scenario)
    // Lane 1: Top, Lane 2: Middle
    for (int j = 0; j < 2; ++j) {
        // Assets use: main_upper = yc - 1.5*lw, main_lower = yc + 0.5*lw
        // Lane 1 center: yc - 1.0 * lw
        // Lane 2 center: yc + 0.0 * lw
        float y = (CY - LW) + j * LW;
        
        std::string in_id = "IN_" + std::to_string(j + 1);
        std::string out_id = "OUT_" + std::to_string(j + 1);

        layout.points[in_id] = {MARGIN, y};
        layout.points[out_id] = {WIDTH - MARGIN, y};
        layout.dir_of[in_id] = "E";
        layout.dir_of[out_id] = "E";
        layout.idx_of[in_id] = j;
        layout.idx_of[out_id] = j;

        layout.in_by_dir["E"].push_back(in_id);
        layout.out_by_dir["E"].push_back(out_id);
    }

    // Ramp (Single lane)
    // Assets use: ramp_start_x = 0, y_center(x) = (yc + 3.8 * lw) + (-0.007 * lw) * x
    // Spawn at x=30: y = yc + (3.8 - 0.007 * 30) * lw = yc + (3.8 - 0.21) * lw = yc + 3.59 * lw
    std::string ramp_in = "IN_RAMP_1";
    layout.points[ramp_in] = {30.0f, CY + 3.59f * LW};
    layout.dir_of[ramp_in] = "R";
    layout.idx_of[ramp_in] = 2; // Treat as 3rd lane index
    layout.in_by_dir["R"].push_back(ramp_in);

    return layout;
}

LaneLayout build_lane_layout_bottleneck_cpp(int num_lanes) {
    LaneLayout layout;
    const float CY = HEIGHT * 0.5f;
    const float MARGIN = 30.0f;

    // Bottleneck: 1D highway-style (W -> E)
    layout.dir_order = {"E"};
    layout.in_by_dir = {{"E", {}}};
    layout.out_by_dir = {{"E", {}}};

    for (int j = 0; j < num_lanes; ++j) {
        float offset = (j - (num_lanes - 1) / 2.0f) * LANE_WIDTH_PX;
        float y_pos = CY + offset;

        std::string in_name = "IN_" + std::to_string(j + 1);
        std::string out_name = "OUT_" + std::to_string(j + 1);

        layout.points[in_name] = {MARGIN, y_pos};
        layout.points[out_name] = {WIDTH - MARGIN, y_pos};
        layout.dir_of[in_name] = "E";
        layout.dir_of[out_name] = "E";
        layout.idx_of[in_name] = j;
        layout.idx_of[out_name] = j;

        layout.in_by_dir["E"].push_back(in_name);
        layout.out_by_dir["E"].push_back(out_name);
    }
    return layout;
}

LaneLayout build_lane_layout_roundabout_cpp(int num_lanes) {
    return build_lane_layout_cpp(num_lanes);
}

int determine_intent(const LaneLayout& layout, const std::string& start_id, const std::string& end_id) {
    auto it_s = layout.dir_of.find(start_id);
    auto it_e = layout.dir_of.find(end_id);
    if (it_s == layout.dir_of.end() || it_e == layout.dir_of.end()) return INTENT_LEFT;
    
    char s = it_s->second[0];
    char e = it_e->second[0];

    auto opposite = [&](char d){
        if (d=='N') return 'S'; if (d=='S') return 'N';
        if (d=='E') return 'W'; return 'E';
    };
    
    // RHT + 环岛逆时针 (CCW) 映射：
    // N -> E 是右转 (1st exit), N -> W 是左转 (3rd exit)
    auto left_turn = [&](char d){
        if (d=='N') return 'W'; if (d=='W') return 'S';
        if (d=='S') return 'E'; if (d=='E') return 'N';
        return 'N';
    };
    auto right_turn = [&](char d){
        if (d=='N') return 'E'; if (d=='E') return 'S';
        if (d=='S') return 'W'; if (d=='W') return 'N';
        return 'N';
    };

    if (e == opposite(s)) return INTENT_STRAIGHT;
    if (e == left_turn(s)) return INTENT_LEFT; 
    if (e == right_turn(s)) return INTENT_RIGHT; 
    return INTENT_LEFT;
}

// ==========================================
// 几何计算：逆时针 (CCW) 模式
// ==========================================
struct ArcDef {
    std::pair<float, float> center;
    float radius;
    float start_angle;
    float end_angle;
};

// 入环圆弧 (RHT + CCW):
// 右侧通行进入环岛是向"右"转并入逆时针车流。
// 圆心位于车辆行驶方向的右侧。
static ArcDef calculate_entry_arc_ccw(std::pair<float, float> pt, std::string dir, 
                                      float ring_r, float fillet_r, float cx, float cy) {
    ArcDef arc;
    arc.radius = fillet_r;
    float R_sum = ring_r + fillet_r; 
    float fx = 0, fy = 0;

    if (dir == "N") { // 向下行驶 (Southbound)。右侧是 -X (West)。
        fx = pt.first - fillet_r; 
        float dx = fx - cx;
        float dy = std::sqrt(std::max(0.0f, R_sum*R_sum - dx*dx));
        fy = cy - dy;
        arc.start_angle = 0.0f; // 指向直线切点 (pt.x, fy)
        arc.end_angle = std::atan2(cy - fy, cx - fx); // 指向与环道的切点
    } 
    else if (dir == "S") { // 向上行驶 (Northbound)。右侧是 +X (East)。
        fx = pt.first + fillet_r;
        float dx = fx - cx;
        float dy = std::sqrt(std::max(0.0f, R_sum*R_sum - dx*dx));
        fy = cy + dy;
        arc.start_angle = PI_F; 
        arc.end_angle = std::atan2(cy - fy, cx - fx);
    }
    else if (dir == "E") { // 向左行驶 (Westbound)。右侧是 -Y (North)。
        fy = pt.second - fillet_r;
        float dy = fy - cy;
        float dx = std::sqrt(std::max(0.0f, R_sum*R_sum - dy*dy));
        fx = cx + dx;
        arc.start_angle = PI_F / 2.0f; 
        arc.end_angle = std::atan2(cy - fy, cx - fx);
    }
    else { // W // 向右行驶 (Eastbound)。右侧是 +Y (South)。
        fy = pt.second + fillet_r;
        float dy = fy - cy;
        float dx = std::sqrt(std::max(0.0f, R_sum*R_sum - dy*dy));
        fx = cx - dx;
        arc.start_angle = -PI_F / 2.0f; 
        arc.end_angle = std::atan2(cy - fy, cx - fx);
    }
    arc.center = std::make_pair(fx, fy);
    return arc;
}

// 出环圆弧 (RHT + CCW):
// 从逆时针圆环切出到直道，同样是向"右"转。
// 圆心位于车辆行驶方向的右侧。
static ArcDef calculate_exit_arc_ccw(std::pair<float, float> pt, std::string dir, 
                                     float ring_r, float fillet_r, float cx, float cy) {
    ArcDef arc;
    arc.radius = fillet_r;
    float R_sum = ring_r + fillet_r;
    float fx = 0, fy = 0;

    if (dir == "N") { // 向上驶出 (Northbound)。右侧是 +X (East)。
        fx = pt.first + fillet_r;
        float dx = fx - cx;
        float dy = std::sqrt(std::max(0.0f, R_sum*R_sum - dx*dx));
        fy = cy - dy;
        arc.start_angle = std::atan2(cy - fy, cx - fx); // 环道切点
        arc.end_angle = PI_F; // 直线切点 (pt.x, fy)
    }
    else if (dir == "S") { // 向下驶出 (Southbound)。右侧是 -X (West)。
        fx = pt.first - fillet_r;
        float dx = fx - cx;
        float dy = std::sqrt(std::max(0.0f, R_sum*R_sum - dx*dx));
        fy = cy + dy;
        arc.start_angle = std::atan2(cy - fy, cx - fx);
        arc.end_angle = 0.0f; 
    }
    else if (dir == "E") { // 向右驶出 (Eastbound)。右侧是 +Y (South)。
        fy = pt.second + fillet_r;
        float dy = fy - cy;
        float dx = std::sqrt(std::max(0.0f, R_sum*R_sum - dy*dy));
        fx = cx + dx;
        arc.start_angle = std::atan2(cy - fy, cx - fx);
        arc.end_angle = -PI_F / 2.0f; 
    }
    else { // W // 向左驶出 (Westbound)。右侧是 -Y (North)。
        fy = pt.second - fillet_r;
        float dy = fy - cy;
        float dx = std::sqrt(std::max(0.0f, R_sum*R_sum - dy*dy));
        fx = cx - dx;
        arc.start_angle = std::atan2(cy - fy, cx - fx);
        arc.end_angle = PI_F / 2.0f; 
    }
    arc.center = std::make_pair(fx, fy);
    return arc;
}

std::vector<std::pair<float,float>> generate_path_bottleneck_cpp(const LaneLayout& layout,
                                                                 int num_lanes,
                                                                 const std::string& start_id,
                                                                 const std::string& end_id) {
    const float CY = HEIGHT * 0.5f;

    // Keep these consistent with scripts/generate_bottleneck_assets.py
    // WIDTH = 1000
    // 左侧直道 X1 = 300
    // 过渡段 TRANS = 150
    // 瓶颈段 BNECK = 100
    const float X1 = 300.0f;
    const float X2 = 450.0f; // 300 + 150
    const float X3 = 550.0f; // 450 + 100
    const float X4 = 700.0f; // 550 + 150

    auto p_start = layout.points.at(start_id);
    auto p_end = layout.points.at(end_id);

    float start_y = p_start.second;
    float end_y = p_end.second;
    float mid_y = CY;

    auto smoothstep = [](float t) {
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        return t * t * (3.0f - 2.0f * t);
    };

    std::vector<std::pair<float,float>> path;
    path.reserve(260);

    const float x0 = p_start.first;
    const float xN = p_end.first;

    const int steps = 240;
    for (int i = 0; i <= steps; ++i) {
        float t = float(i) / float(steps);
        float x = x0 + (xN - x0) * t;

        float y;
        if (x <= X1) {
            y = start_y;
        } else if (x <= X2) {
            float lt = (x - X1) / (X2 - X1);
            y = start_y + (mid_y - start_y) * smoothstep(lt);
        } else if (x <= X3) {
            y = mid_y;
        } else if (x <= X4) {
            float lt = (x - X3) / (X4 - X3);
            y = mid_y + (end_y - mid_y) * smoothstep(lt);
        } else {
            y = end_y;
        }

        path.emplace_back(x, y);
    }

    return path;
}

std::vector<std::pair<float,float>> generate_path_roundabout_cpp(const LaneLayout& layout,
                                                                 int num_lanes,
                                                                 int intent,
                                                                 const std::string& start_id,
                                                                 const std::string& end_id) {
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;
    const float INNER_RADIUS = 150.0f;
    
    int start_lane_idx = layout.idx_of.at(start_id);
    
    // ==========================================
    // 逆时针 (CCW) 车道选择逻辑
    // ==========================================
    // 0 = Inner (靠近圆心), num_lanes-1 = Outer (靠近外围)
    
    int ring_lane_idx = start_lane_idx;
    
    // CCW规则：
    // 右转 (1st exit): 是最近的出口，走最外侧 (Outer Lane)
    // 左转 (3rd exit): 绕行大半圈，走最内侧 (Inner Lane)
    if (intent == INTENT_RIGHT) {
        ring_lane_idx = num_lanes - 1; 
    } else if (intent == INTENT_LEFT) {
        ring_lane_idx = 0; 
    }

    if (ring_lane_idx < 0) ring_lane_idx = 0;
    if (ring_lane_idx >= num_lanes) ring_lane_idx = num_lanes - 1;

    float ring_r = INNER_RADIUS + (float(ring_lane_idx) + 0.5f) * LANE_WIDTH_PX;
    float fillet_r = FILLET_RADIUS_BASE + (float(start_lane_idx) + 0.5f) * LANE_WIDTH_PX;

    auto p_start = layout.points.at(start_id);
    auto p_end = layout.points.at(end_id);
    std::string start_dir = layout.dir_of.at(start_id);
    std::string end_dir = layout.dir_of.at(end_id);

    std::vector<std::pair<float,float>> path;
    path.reserve(500);

    // ==========================================
    // Phase 1: Entry (Curves Left/CCW)
    // ==========================================
    ArcDef entry_arc = calculate_entry_arc_ccw(p_start, start_dir, ring_r, fillet_r, CX, CY);

    std::pair<float, float> entry_tangent;
    if (start_dir == "N" || start_dir == "S") 
        entry_tangent = std::make_pair(p_start.first, entry_arc.center.second);
    else 
        entry_tangent = std::make_pair(entry_arc.center.first, p_start.second);

    for(int i=0; i<=20; ++i) {
        float t = float(i)/20.0f;
        path.emplace_back(
            p_start.first + (entry_tangent.first - p_start.first)*t,
            p_start.second + (entry_tangent.second - p_start.second)*t
        );
    }

    float da_in = normalize_angle(entry_arc.end_angle - entry_arc.start_angle);
    // 入环是向左转，角度增量应符合逆时针方向。
    // 在屏幕坐标系(Y下)中：
    // N入口(start PI) -> 左转(逆时针) -> end angle (e.g. PI + delta)
    // 逆时针意味着角度数值可能增大？(X轴0 -> Y轴PI/2)。
    // 让我们确保 da_in 为最短路径。
    if (da_in > PI_F) da_in -= 2*PI_F; 
    if (da_in < -PI_F) da_in += 2*PI_F;

    int steps_in = std::max(10, int(std::abs(da_in) * 40.0f));
    for(int i=1; i<=steps_in; ++i) {
        float t = float(i)/steps_in;
        float ang = entry_arc.start_angle + da_in * t;
        path.emplace_back(
            entry_arc.center.first + entry_arc.radius * std::cos(ang),
            entry_arc.center.second + entry_arc.radius * std::sin(ang)
        );
    }

    // ==========================================
    // Phase 2: Ring (Counter-Clockwise)
    // ==========================================
    float ring_ang_start = std::atan2(path.back().second - CY, path.back().first - CX);

    int end_lane_idx = layout.idx_of.at(end_id);
    float exit_fillet_r = FILLET_RADIUS_BASE + (float(end_lane_idx) + 0.5f) * LANE_WIDTH_PX;
    ArcDef exit_arc = calculate_exit_arc_ccw(p_end, end_dir, ring_r, exit_fillet_r, CX, CY);

    float ring_ang_end = std::atan2(exit_arc.center.second - CY, exit_arc.center.first - CX);

    float da_ring = ring_ang_end - ring_ang_start;
    // 逆时针(CCW): 
    // East(0) -> North(-PI/2) -> West(PI) -> South(PI/2)
    // 角度应该是减小的 (e.g. 0 -> -1.57)。
    // 所以 delta 应该是负数。
    while (da_ring > 0) da_ring -= 2.0f * PI_F;

    if (std::abs(da_ring) < 1e-4) da_ring = 0;

    int steps_ring = std::max(20, int(std::abs(da_ring) * 30.0f));
    for(int i=1; i<=steps_ring; ++i) {
        float t = float(i)/steps_ring;
        float ang = ring_ang_start + da_ring * t;
        path.emplace_back(
            CX + ring_r * std::cos(ang),
            CY + ring_r * std::sin(ang)
        );
    }

    // ==========================================
    // Phase 3: Exit (Curves Right out)
    // ==========================================
    float da_out = normalize_angle(exit_arc.end_angle - exit_arc.start_angle);
    if (da_out > PI_F) da_out -= 2*PI_F;
    if (da_out < -PI_F) da_out += 2*PI_F;

    int steps_out = std::max(10, int(std::abs(da_out) * 40.0f));
    for(int i=1; i<=steps_out; ++i) {
        float t = float(i)/steps_out;
        float ang = exit_arc.start_angle + da_out * t;
        path.emplace_back(
            exit_arc.center.first + exit_arc.radius * std::cos(ang),
            exit_arc.center.second + exit_arc.radius * std::sin(ang)
        );
    }

    std::pair<float,float> exit_start = path.back();
    for(int i=1; i<=20; ++i) {
        float t = float(i)/20.0f;
        path.emplace_back(
            exit_start.first + (p_end.first - exit_start.first)*t,
            exit_start.second + (p_end.second - exit_start.second)*t
        );
    }
    return path;
}

std::vector<std::pair<float,float>> generate_path_cpp(const LaneLayout& layout,
                                                      int num_lanes,
                                                      int intent,
                                                      const std::string& start_id,
                                                      const std::string& end_id) {
    const float CX = WIDTH * 0.5f;
    const float CY = HEIGHT * 0.5f;

    auto p_start = layout.points.at(start_id);
    auto p_end = layout.points.at(end_id);
    
    std::string start_dir = layout.dir_of.at(start_id);
    std::string end_dir = layout.dir_of.at(end_id);

    auto entry_p = get_lane_dash_end_point(p_start, start_dir, num_lanes);
    auto exit_p = get_lane_dash_end_point(p_end, end_dir, num_lanes);

    std::vector<std::pair<float,float>> path;
    path.reserve(200);

    // Special-case: Merge scenario ramp path (bitmap-based)
    // IN_RAMP_1 -> OUT_2: ramp into 3-lane section, then merge into lane2 near the narrowing fillet.
    if (start_id == "IN_RAMP_1" && end_id == "OUT_2") {
        const float lw = LANE_WIDTH_PX;
        const float y_lane2 = CY;                         // center of lane 2
        const float y_lane3 = CY + 1.0f * lw;             // center of added bottom lane (lane 3)

        // Assets coordinates
        const float x_spawn = 30.0f;
        const float x_merge = 400.0f;
        const float x_drop = 800.0f;
        
        // y centers at specific x positions based on asset slope
        // At x=0, y_center = CY + 3.8*lw. At x=400, y_center = CY + 1.0*lw
        // slope k = (1.0 - 3.8) * lw / 400 = -0.007 * lw
        auto get_ramp_y = [&](float x) {
            float k = -0.007f * lw;
            return (CY + 3.8f * lw) + k * x;
        };

        // 1) Ramp internal path (from x=30 to x=merge, strictly following ramp center)
        const int steps1 = 100;
        for (int i = 0; i <= steps1; ++i) {
            float t = (float)i / steps1;
            float x = x_spawn + (x_merge - x_spawn) * t;
            path.emplace_back(x, get_ramp_y(x));
        }

        // 2) Stay in Lane 3 (already at y_lane3 at x_merge)
        const float x_curve2_start = x_drop - 150.0f;
        if (x_curve2_start > x_merge) {
            const int steps3 = 40;
            for (int i = 1; i <= steps3; ++i) {
                float t = (float)i / steps3;
                float x = x_merge + (x_curve2_start - x_merge) * t;
                path.emplace_back(x, y_lane3);
            }
        }

        // 4) Smooth curve from Lane 3 into Lane 2 (before Lane 3 ends)
        const float x_curve2_end = x_drop - 20.0f;
        const int steps4 = 50;
        for (int i = 1; i <= steps4; ++i) {
            float t = (float)i / steps4;
            float x = x_curve2_start + (x_curve2_end - x_curve2_start) * t;
            float s = 0.5f - 0.5f * std::cos(PI_F * t); // smoothstep
            float y = y_lane3 + (y_lane2 - y_lane3) * s;
            path.emplace_back(x, y);
        }

        // 5) Final straight to OUT_2
        const int steps5 = 40;
        const float x_final = p_end.first;
        for (int i = 1; i <= steps5; ++i) {
            float t = (float)i / steps5;
            float x = x_curve2_end + (x_final - x_curve2_end) * t;
            path.emplace_back(x, y_lane2);
        }

        return path;
    }

    if (intent == INTENT_STRAIGHT) { 
        for (int i=0; i<=40; ++i){
            float t=float(i)/40.0f;
            path.emplace_back(
                p_start.first + (entry_p.first-p_start.first)*t,
                p_start.second + (entry_p.second-p_start.second)*t
            );
        }
        for (int i=1; i<=60; ++i){
            float t=float(i)/60.0f;
            path.emplace_back(
                entry_p.first + (exit_p.first-entry_p.first)*t,
                entry_p.second + (exit_p.second-entry_p.second)*t
            );
        }
        for (int i=1; i<=40; ++i){
            float t=float(i)/40.0f;
            path.emplace_back(
                exit_p.first + (p_end.first-exit_p.first)*t,
                exit_p.second + (p_end.second-exit_p.second)*t
            );
        }
        return path;
    } 

    for (int i=0; i<=40; ++i){
        float t=float(i)/40.0f;
        path.emplace_back(
            p_start.first + (entry_p.first-p_start.first)*t,
            p_start.second + (entry_p.second-p_start.second)*t
        );
    }

    // Turn segment: circular arc from entry stop line to exit stop line
    std::pair<float,float> center;
    if ((start_dir == "N" || start_dir == "S") && (end_dir == "E" || end_dir == "W")) {
        center = std::make_pair(exit_p.first, entry_p.second);
    } else {
        center = std::make_pair(entry_p.first, exit_p.second);
    }

    float r = std::sqrt((entry_p.first-center.first)*(entry_p.first-center.first) + 
                        (entry_p.second-center.second)*(entry_p.second-center.second));
    float a_start = std::atan2(entry_p.second - center.second, entry_p.first - center.first);
    float a_end = std::atan2(exit_p.second - center.second, exit_p.first - center.first);

    float da = a_end - a_start;
    while (da > PI_F) da -= 2.0f * PI_F;
    while (da < -PI_F) da += 2.0f * PI_F;

    // RHT: right turn should take the short, inner arc; left turn should take the longer, outer arc
    if (intent == INTENT_LEFT) {
        if (da > 0) da -= 2.0f * PI_F;
    } else if (intent == INTENT_RIGHT) {
        if (da < 0) da += 2.0f * PI_F;
    }

    int steps = 100;
    for (int i=1; i<=steps; ++i) {
        float t = float(i)/float(steps);
        float angle = a_start + da * t;
        path.emplace_back(center.first + r * std::cos(angle), center.second + r * std::sin(angle));
    }

    for (int i=1; i<=40; ++i){
        float t=float(i)/40.0f;
        path.emplace_back(
            exit_p.first + (p_end.first-exit_p.first)*t,
            exit_p.second + (p_end.second-exit_p.second)*t
        );
    }
    return path;
}