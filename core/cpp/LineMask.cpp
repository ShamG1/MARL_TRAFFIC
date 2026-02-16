#include "LineMask.h"
#include <algorithm>

LineMask::LineMask(int num_lanes) {
    grid.assign(size_t(width) * size_t(height), uint8_t(0));
    generate(num_lanes);
}

static inline void set_px(std::vector<uint8_t>& g, int w, int h, int x, int y, uint8_t v) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    g[size_t(y) * size_t(w) + size_t(x)] = v;
}

void LineMask::draw_thick_line(std::vector<uint8_t>& g, int w, int h,
                              int x0, int y0, int x1, int y1,
                              int thickness, uint8_t v) {
    // 仅支持水平/垂直（与 env.py 的 line_mask 一致）
    const int half = std::max(0, thickness / 2);

    if (x0 == x1) {
        int x = x0;
        int ya = std::min(y0, y1);
        int yb = std::max(y0, y1);
        for (int y = ya; y <= yb; ++y) {
            for (int dx = -half; dx <= half; ++dx) {
                set_px(g, w, h, x + dx, y, v);
            }
        }
        return;
    }

    if (y0 == y1) {
        int y = y0;
        int xa = std::min(x0, x1);
        int xb = std::max(x0, x1);
        for (int x = xa; x <= xb; ++x) {
            for (int dy = -half; dy <= half; ++dy) {
                set_px(g, w, h, x, y + dy, v);
            }
        }
        return;
    }

    // fallback：非轴对齐就不画（当前不需要）
}

void LineMask::generate(int num_lanes) {
    // 复刻 Scenario/env.py::Road._generate_line_mask
    // 白线=1，背景=0
    const int cx = WIDTH / 2;
    const int cy = HEIGHT / 2;
    const int rw = int(num_lanes * int(LANE_WIDTH_PX));
    const int cr = int(CORNER_RADIUS);
    const int stop_offset = rw + cr;

    std::fill(grid.begin(), grid.end(), uint8_t(0));

    // 对齐 pygame.draw.line(..., width=2)
    const int line_w = 2;

    // Vertical Lines
    draw_thick_line(grid, width, height, cx - 2, 0, cx - 2, cy - stop_offset, line_w, 1);
    draw_thick_line(grid, width, height, cx + 2, 0, cx + 2, cy - stop_offset, line_w, 1);
    draw_thick_line(grid, width, height, cx - 2, height, cx - 2, cy + stop_offset, line_w, 1);
    draw_thick_line(grid, width, height, cx + 2, height, cx + 2, cy + stop_offset, line_w, 1);

    // Horizontal Lines
    draw_thick_line(grid, width, height, 0, cy - 2, cx - stop_offset, cy - 2, line_w, 1);
    draw_thick_line(grid, width, height, 0, cy + 2, cx - stop_offset, cy + 2, line_w, 1);
    draw_thick_line(grid, width, height, width, cy - 2, cx + stop_offset, cy - 2, line_w, 1);
    draw_thick_line(grid, width, height, width, cy + 2, cx + stop_offset, cy + 2, line_w, 1);
}
