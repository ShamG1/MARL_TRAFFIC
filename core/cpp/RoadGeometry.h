#pragma once
#include "constants.h"
#include <cmath>
#include <array>

class RoadGeometry {
public:
    int lanes;
    float lane_w;
    float half_road_w;
    float corner_r;

    explicit RoadGeometry(int num_lanes = 3)
        : lanes(num_lanes),
          lane_w(LANE_WIDTH_PX),
          half_road_w(num_lanes * LANE_WIDTH_PX),
          corner_r(CORNER_RADIUS) {}

    inline bool is_on_road(float x, float y) const {
        const float CX = WIDTH * 0.5f;
        const float CY = HEIGHT * 0.5f;

        // Exactly match Renderer::draw_road():
        // road = (vertical strip ∪ horizontal strip ∪ 4 corner squares) \ 4 grass circles
        const float rw = half_road_w;
        const float cr = corner_r;

        // 1) Subtract grass circles first (inside circle => off-road)
        const float r2 = cr * cr;
        const std::array<std::pair<float,float>,4> grass_centers = {{
            {CX - rw - cr, CY - rw - cr},
            {CX + rw + cr, CY - rw - cr},
            {CX - rw - cr, CY + rw + cr},
            {CX + rw + cr, CY + rw + cr}
        }};
        for(const auto& c : grass_centers){
            const float dx = x - c.first;
            const float dy = y - c.second;
            if(dx*dx + dy*dy <= r2) return false;
        }

        // 2) Main cross surface (two rectangles spanning the full screen)
        const bool in_vertical_strip = (x >= CX - rw) && (x <= CX + rw);
        const bool in_horizontal_strip = (y >= CY - rw) && (y <= CY + rw);
        if(in_vertical_strip || in_horizontal_strip) return true;

        // 3) Corner squares (cr x cr) filled as road
        // top-left
        if(x >= CX - rw - cr && x <= CX - rw && y >= CY - rw - cr && y <= CY - rw) return true;
        // top-right
        if(x >= CX + rw && x <= CX + rw + cr && y >= CY - rw - cr && y <= CY - rw) return true;
        // bottom-left
        if(x >= CX - rw - cr && x <= CX - rw && y >= CY + rw && y <= CY + rw + cr) return true;
        // bottom-right
        if(x >= CX + rw && x <= CX + rw + cr && y >= CY + rw && y <= CY + rw + cr) return true;

        return false;
    }

    inline bool hits_yellow_line(float x, float y) const {
        float cx = WIDTH * 0.5f;
        float cy = HEIGHT * 0.5f;
        float gap = 2.0f;
        if (std::fabs(x - cx) <= gap && std::fabs(y - cy) > half_road_w) return true;
        if (std::fabs(y - cy) <= gap && std::fabs(x - cx) > half_road_w) return true;
        return false;
    }
};
