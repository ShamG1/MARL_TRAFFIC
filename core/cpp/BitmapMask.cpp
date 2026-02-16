#include "BitmapMask.h"

#include "stb_image.h"

#include <algorithm>

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
