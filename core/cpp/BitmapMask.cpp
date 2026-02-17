#include "BitmapMask.h"

#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <limits>

bool BitmapMask::load_grayscale_png(const std::string& path) {
    int w = 0;
    int h = 0;
    int c = 0;

    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!raw || w <= 0 || h <= 0 || c <= 0) {
        if (raw) stbi_image_free(raw);
        width = height = channels = 0;
        data.clear();
        return false;
    }

    width = w;
    height = h;
    channels = c;
    data.assign(size_t(width) * size_t(height), uint8_t(0));

    // Convert to grayscale using a simple luminance approximation.
    // Accepts 1/2/3/4 channels.
    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            const size_t idx = (size_t(yy) * size_t(width) + size_t(xx));
            const size_t base = idx * size_t(channels);

            uint8_t g = 0;
            if (channels == 1) {
                g = raw[base];
            } else {
                const uint8_t r = raw[base + 0];
                const uint8_t gg = raw[base + 1];
                const uint8_t b = raw[base + 2];
                // 0.299R + 0.587G + 0.114B
                g = uint8_t((77 * int(r) + 150 * int(gg) + 29 * int(b)) >> 8);
            }
            data[idx] = g;
        }
    }

    stbi_image_free(raw);
    return true;
}

void BitmapMask::compute_sdf() {
    if (!loaded()) return;

    size_t n = size_t(width) * size_t(height);
    sdf_data.assign(n, std::numeric_limits<float>::max());

    // Meijster's algorithm for Euclidean Distance Transform
    // Pass 1: scan columns
    std::vector<float> g(n, 1e10f);
    for (int xx = 0; xx < width; ++xx) {
        // Vertical pass
        if (at(xx, 0) == 0) g[xx] = 0;
        for (int yy = 1; yy < height; ++yy) {
            size_t idx = size_t(yy) * size_t(width) + size_t(xx);
            if (at(xx, yy) == 0) g[idx] = 0;
            else g[idx] = 1 + g[idx - size_t(width)];
        }
        for (int yy = height - 2; yy >= 0; --yy) {
            size_t idx = size_t(yy) * size_t(width) + size_t(xx);
            if (g[idx + size_t(width)] + 1 < g[idx]) {
                g[idx] = 1 + g[idx + size_t(width)];
            }
        }
    }

    // Pass 2: scan rows
    auto f = [&](int x, int i, float gi) -> float {
        return float((x - i) * (x - i)) + gi * gi;
    };

    auto sep = [&](int i, int j, float gi, float gj) -> int {
        return int((float(j * j - i * i) + gj * gj - gi * gi) / (2.0f * float(j - i)));
    };

    std::vector<int> s(width);
    std::vector<int> t(width);

    for (int yy = 0; yy < height; ++yy) {
        int q = 0;
        s[0] = 0;
        t[0] = 0;

        for (int u = 1; u < width; ++u) {
            size_t idx_u = size_t(yy) * size_t(width) + size_t(u);
            float gu = g[idx_u];
            
            while (q >= 0) {
                size_t idx_sq = size_t(yy) * size_t(width) + size_t(s[q]);
                if (f(t[q], u, gu) > f(t[q], s[q], g[idx_sq])) break;
                q--;
            }

            if (q < 0) {
                q = 0;
                s[0] = u;
                t[0] = 0;
            } else {
                size_t idx_sq = size_t(yy) * size_t(width) + size_t(s[q]);
                int w = 1 + sep(s[q], u, g[idx_sq], gu);
                if (w < width) {
                    q++;
                    s[q] = u;
                    t[q] = w;
                }
            }
        }

        for (int u = width - 1; u >= 0; --u) {
            size_t idx_sq = size_t(yy) * size_t(width) + size_t(s[q]);
            sdf_data[size_t(yy) * size_t(width) + size_t(u)] = std::sqrt(f(u, s[q], g[idx_sq]));
            if (u == t[q]) q--;
        }
    }
}
