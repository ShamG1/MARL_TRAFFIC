#include "Lidar.h"
#include <algorithm>

Lidar::Lidar() {
    distances.assign(rays, max_dist);
    rel_angles.clear();
    const float start_angle_deg = -fov_deg * 0.5f;
    const float step_deg = (rays > 1) ? (fov_deg / float(rays - 1)) : 0.0f;
    constexpr float PI_F = 3.14159265358979323846f;
    for (int i = 0; i < rays; ++i) {
        float deg = start_angle_deg + i * step_deg;
        rel_angles.push_back(deg * PI_F / 180.0f);
    }
}

void Lidar::update(const Car& self, const std::vector<Car>& cars, const RoadGeometry& geom,
                   int width, int height) {
    if ((int)distances.size() != rays) distances.assign(rays, max_dist);

    const float cx = self.state.x;
    const float cy = self.state.y;
    const float heading = self.state.heading;

    for (int i = 0; i < rays; ++i) {
        const float ray_angle = heading + rel_angles[i];
        const float dx = std::cos(ray_angle);
        const float dy = -std::sin(ray_angle); // match Scenario/sensor.py

        bool hit_found = false;
        float final_dist = max_dist;

        // Match Scenario/sensor.py: start from 0
        for (float dist = 0.0f; dist < max_dist; dist += step_size) {
            int check_x = int(cx + dx * dist);
            int check_y = int(cy + dy * dist);

            // 1) screen boundary: shot into void
            if (check_x < 0 || check_x >= width || check_y < 0 || check_y >= height) {
                break;
            }

            // 2) static road detection: off-road is obstacle (rounded corners via RoadGeometry)
            // Skip dist==0 to avoid self-point aliasing turning every ray into an immediate hit
            if (dist > 0.0f && !geom.is_on_road(float(check_x), float(check_y))) {
                hit_found = true;
                final_dist = dist;
                break;
            }

            // 3) dynamic vehicle detection
            if (dist > 0.0f) {
                bool collision = false;
                for (const auto& c : cars) {
                    // NOTE: `cars` may include a copy of `self`.
                    if (&c == &self) continue;
                    if (std::fabs(c.state.x - self.state.x) < 1e-3f &&
                        std::fabs(c.state.y - self.state.y) < 1e-3f &&
                        std::fabs(c.state.heading - self.state.heading) < 1e-3f) {
                        continue;
                    }

                    const float cosA = std::cos(c.state.heading);
                    const float sinA = std::sin(c.state.heading);
                    const float hl = c.length * 0.5f;
                    const float hw = c.width * 0.5f;

                    const float ex = std::fabs(cosA) * hl + std::fabs(sinA) * hw;
                    const float ey = std::fabs(sinA) * hl + std::fabs(cosA) * hw;

                    if (float(check_x) >= c.state.x - ex && float(check_x) <= c.state.x + ex &&
                        float(check_y) >= c.state.y - ey && float(check_y) <= c.state.y + ey) {
                        collision = true;
                        break;
                    }
                }
                if (collision) {
                    hit_found = true;
                    final_dist = dist;
                    break;
                }
            }
        }

        distances[i] = hit_found ? final_dist : max_dist;
    }
}

void Lidar::update_bitmap(const Car& self, const std::vector<Car>& cars, const BitmapMask& road_mask,
                          int width, int height) {
    if ((int)distances.size() != rays) distances.assign(rays, max_dist);

    const float cx = self.state.x;
    const float cy = self.state.y;
    const float heading = self.state.heading;

    for (int i = 0; i < rays; ++i) {
        const float ray_angle = heading + rel_angles[i];
        const float dx = std::cos(ray_angle);
        const float dy = -std::sin(ray_angle);

        bool hit_found = false;
        float final_dist = max_dist;

        for (float dist = 0.0f; dist < max_dist; dist += step_size) {
            int check_x = int(cx + dx * dist);
            int check_y = int(cy + dy * dist);

            if (check_x < 0 || check_x >= width || check_y < 0 || check_y >= height) {
                break;
            }

            // Static obstacle: anything not drivable on the road mask.
            if (dist > 0.0f && road_mask.at(check_x, check_y) == 0) {
                hit_found = true;
                final_dist = dist;
                break;
            }

            // Dynamic vehicle detection (same as geom mode)
            if (dist > 0.0f) {
                bool collision = false;
                for (const auto& c : cars) {
                    if (&c == &self) continue;
                    if (std::fabs(c.state.x - self.state.x) < 1e-3f &&
                        std::fabs(c.state.y - self.state.y) < 1e-3f &&
                        std::fabs(c.state.heading - self.state.heading) < 1e-3f) {
                        continue;
                    }

                    const float cosA = std::cos(c.state.heading);
                    const float sinA = std::sin(c.state.heading);
                    const float hl = c.length * 0.5f;
                    const float hw = c.width * 0.5f;

                    const float ex = std::fabs(cosA) * hl + std::fabs(sinA) * hw;
                    const float ey = std::fabs(sinA) * hl + std::fabs(cosA) * hw;

                    if (float(check_x) >= c.state.x - ex && float(check_x) <= c.state.x + ex &&
                        float(check_y) >= c.state.y - ey && float(check_y) <= c.state.y + ey) {
                        collision = true;
                        break;
                    }
                }
                if (collision) {
                    hit_found = true;
                    final_dist = dist;
                    break;
                }
            }
        }

        distances[i] = hit_found ? final_dist : max_dist;
    }
}

std::vector<float> Lidar::normalized() const {
    std::vector<float> out;
    out.reserve(distances.size());
    const float inv = (max_dist > 0.0f) ? (1.0f / max_dist) : 0.0f;
    for (float d : distances) out.push_back(d * inv);
    return out;
}
