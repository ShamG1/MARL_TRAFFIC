#pragma once
#include <vector>
#include <cmath>
#include "Car.h"
#include "constants.h"
#include "RoadGeometry.h"
#include "BitmapMask.h"

class Lidar {
public:
    // Match Scenario/config.py
    int rays{72};
    float fov_deg{360.0f};
    float max_dist{250.0f};
    float step_size{1.0f};

    std::vector<float> distances;
    std::vector<float> rel_angles; // radians

    Lidar();

    // Update for a given car; off-road is treated as obstacle via RoadGeometry.
    // Accepts two car lists (e.g., egos and NPCs) to avoid temporary vector merging.
    void update(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2,
                const RoadGeometry& geom, int width = WIDTH, int height = HEIGHT);

    // Update for bitmap scenarios; road mask defines free space (at(x,y)>0 is drivable).
    void update_bitmap(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2,
                       const BitmapMask& road_mask, int width = WIDTH, int height = HEIGHT);

    // Normalized readings (dist/max_dist)
    std::vector<float> normalized() const;
};
