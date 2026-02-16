#pragma once
#include <cstdint>
#include <vector>
#include "constants.h"

// 像素级 road/collision mask：与 Scenario/env.py 里的 Road._generate_collision_mask 等价
// 约定：1 = obstacle(白色, 草地/墙)，0 = road(黑色)
class RoadMask {
public:
    int width{WIDTH};
    int height{HEIGHT};

    explicit RoadMask(int num_lanes = 3);

    inline bool is_obstacle(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) return false; // 出界按 Python：射线直接 break，不算命中
        return grid[size_t(y) * size_t(width) + size_t(x)] != 0;
    }

private:
    std::vector<uint8_t> grid;

    void generate(int num_lanes);
    static inline void fill_rect(std::vector<uint8_t>& g, int w, int h, int x, int y, int rw, int rh, uint8_t v);
    static inline void fill_circle(std::vector<uint8_t>& g, int w, int h, int cx, int cy, int r, uint8_t v);
};
