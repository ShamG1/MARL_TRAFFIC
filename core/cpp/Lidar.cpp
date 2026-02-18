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

void Lidar::update(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2,
                   const RoadGeometry& geom, int width, int height) {
    if ((int)distances.size() != rays) distances.assign(rays, max_dist);

    const float cx = self.state.x;
    const float cy = self.state.y;
    const float heading = self.state.heading;

    auto check_cars = [&](const std::vector<Car>& car_list, float check_x, float check_y) {
        for (const auto& c : car_list) {
                    if (&c == &self) continue;
            // Additional logical check to skip self even if passed as a copy
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
                return true;
                }
            }
        return false;
    };

    for (int i = 0; i < rays; ++i) {
        const float ray_angle = heading + rel_angles[i];
        const float dx = std::cos(ray_angle);
        const float dy = -std::sin(ray_angle);

        bool hit_found = false;
        float final_dist = max_dist;

        for (float dist = 0.0f; dist < max_dist; dist += step_size) {
            float check_xf = cx + dx * dist;
            float check_yf = cy + dy * dist;
            int check_x = int(check_xf);
            int check_y = int(check_yf);

            if (check_x < 0 || check_x >= width || check_y < 0 || check_y >= height) {
                break;
            }

            if (dist > 0.0f && !geom.is_on_road(float(check_x), float(check_y))) {
                hit_found = true;
                final_dist = dist;
                break;
            }

            if (dist > 0.0f) {
                if (check_cars(cars1, float(check_x), float(check_y)) || check_cars(cars2, float(check_x), float(check_y))) {
                    hit_found = true;
                    final_dist = dist;
                    break;
                }
            }
        }
        distances[i] = hit_found ? final_dist : max_dist;
    }
}

void Lidar::update_bitmap(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2,
                          const BitmapMask& road_mask, int width, int height) {
    if ((int)distances.size() != rays) distances.assign(rays, max_dist);

    const float cx = self.state.x;
    const float cy = self.state.y;
    const float heading = self.state.heading;

    // Dynamic obstacle check: cheap radius filter first, then oriented bbox approx.
    // Uses squared distances to avoid sqrt in the inner loop.
    constexpr float kCheckRadius = 140.0f;
    constexpr float kCheckRadius2 = kCheckRadius * kCheckRadius;

    auto check_cars = [&](const std::vector<Car>& car_list, float check_x, float check_y) {
        for (const auto& c : car_list) {
            if (&c == &self) continue;
            if (std::fabs(c.state.x - self.state.x) < 1e-3f &&
                std::fabs(c.state.y - self.state.y) < 1e-3f &&
                std::fabs(c.state.heading - self.state.heading) < 1e-3f) {
                continue;
            }

            const float dx0 = c.state.x - check_x;
            const float dy0 = c.state.y - check_y;
            if (dx0 * dx0 + dy0 * dy0 > kCheckRadius2) continue;

            const float cosA = std::cos(c.state.heading);
            const float sinA = std::sin(c.state.heading);
            const float hl = c.length * 0.5f;
            const float hw = c.width * 0.5f;

            const float ex = std::fabs(cosA) * hl + std::fabs(sinA) * hw;
            const float ey = std::fabs(sinA) * hl + std::fabs(cosA) * hw;

            if (check_x >= c.state.x - ex && check_x <= c.state.x + ex &&
                check_y >= c.state.y - ey && check_y <= c.state.y + ey) {
                return true;
            }
        }
        return false;
    };

    for (int i = 0; i < rays; ++i) {
        const float ray_angle = heading + rel_angles[i];
        const float dx = std::cos(ray_angle);
        const float dy = -std::sin(ray_angle);

        bool hit_found = false;
        float final_dist = max_dist;

        float curr_dist = 0.0f;
        while (curr_dist < max_dist) {
            int ix = int(cx + dx * curr_dist);
            int iy = int(cy + dy * curr_dist);

            if (ix < 0 || ix >= width || iy < 0 || iy >= height) break;

            float static_dist = road_mask.get_dist(ix, iy);

            if (curr_dist > 0.0f && static_dist < 1.0f) {
                hit_found = true;
                final_dist = curr_dist;
                break;
            }

            if (curr_dist > 0.0f && (check_cars(cars1, float(ix), float(iy)) || check_cars(cars2, float(ix), float(iy)))) {
                hit_found = true;
                final_dist = curr_dist;
                break;
            }

            float jump = std::max(step_size, static_dist);
            jump = std::min(jump, 12.0f); 
            curr_dist += jump;
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
