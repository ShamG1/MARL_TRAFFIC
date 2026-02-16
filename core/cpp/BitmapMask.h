#pragma once
#include <cstdint>
#include <string>
#include <vector>

class BitmapMask {
public:
    int width{0};
    int height{0};
    int channels{0};

    // grayscale data in row-major, size = width*height
    std::vector<uint8_t> data;

    bool loaded() const { return width > 0 && height > 0 && !data.empty(); }

    // Load an image file and convert to 8-bit grayscale.
    // Returns true on success.
    bool load_grayscale_png(const std::string& path);

    inline uint8_t at(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0;
        return data[size_t(y) * size_t(width) + size_t(x)];
    }
};
