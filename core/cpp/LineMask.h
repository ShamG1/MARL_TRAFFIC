#pragma once
#include <cstdint>
#include <vector>
#include "constants.h"

// 像素级黄线/车道线 mask：与 Scenario/env.py 里的 Road._generate_line_mask 等价
// 约定：1 = line(白色)，0 = background(黑色)
class LineMask {
public:
    int width{WIDTH};
    int height{HEIGHT};

    explicit LineMask(int num_lanes = 3);

    inline bool is_line(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) return false;
        return grid[size_t(y) * size_t(width) + size_t(x)] != 0;
    }

private:
    std::vector<uint8_t> grid;

    void generate(int num_lanes);
    static inline void draw_thick_line(std::vector<uint8_t>& g, int w, int h,
                                       int x0, int y0, int x1, int y1,
                                       int thickness, uint8_t v);
};
