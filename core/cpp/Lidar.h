#pragma once
#include <vector>
#include <cmath>
#include "Car.h"
#include "constants.h"
#include "RoadGeometry.h"
#include "BitmapMask.h"

class Lidar {
public:
    struct CachedCar {
        float x, y;
        float cosH, sinH;
        float hl, hw;
        float radius_sq;
    };

    // Match Scenario/config.py
    int rays{72};
    float fov_deg{360.0f};
    float max_dist{250.0f};
    float step_size{1.0f};

    std::vector<float> distances;
    std::vector<float> rel_angles; // radians

    // Precomputed trig for rel_angles (same length as rel_angles)
    std::vector<float> rel_cos;
    std::vector<float> rel_sin;

    Lidar();

    // Update for a given car; off-road is treated as obstacle via RoadGeometry.
    // Accepts two car lists (e.g., egos and NPCs) to avoid temporary vector merging.
    void update(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2,
                const RoadGeometry& geom, int width = WIDTH, int height = HEIGHT);

    // Update for bitmap scenarios; road mask defines free space (at(x,y)>0 is drivable).
    void update_bitmap(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2,
                       const BitmapMask& road_mask, int width = WIDTH, int height = HEIGHT);

    // B1: Efficient updates using externally built obstacle cache
    void update_from_cache(const Car& self, const std::vector<CachedCar>& obstacles,
                           const RoadGeometry& geom, int width = WIDTH, int height = HEIGHT);
    void update_bitmap_from_cache(const Car& self, const std::vector<CachedCar>& obstacles,
                       const BitmapMask& road_mask, int width = WIDTH, int height = HEIGHT);

    // Normalized readings (dist/max_dist)
    // Avoid per-step allocations by writing into a caller-provided buffer.
    void normalized_into(float* out, int out_len) const;

    // Legacy API (allocates). Kept for compatibility.
    std::vector<float> normalized() const;

private:
    void fill_cache(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2) const;

    // Cache to avoid per-update allocations.
    mutable std::vector<CachedCar> cache_;
};
