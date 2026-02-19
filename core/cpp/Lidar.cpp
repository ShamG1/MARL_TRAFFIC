#include "Lidar.h"
#include <algorithm>

Lidar::Lidar() {
    distances.assign(rays, max_dist);
    rel_angles.clear();
    const float start_angle_deg = -fov_deg * 0.5f;
    const float step_deg = (rays > 1) ? (fov_deg / float(rays - 1)) : 0.0f;
    constexpr float PI_F = 3.14159265358979323846f;
    rel_cos.clear();
    rel_sin.clear();
    rel_cos.reserve(rays);
    rel_sin.reserve(rays);

    for (int i = 0; i < rays; ++i) {
        float deg = start_angle_deg + i * step_deg;
        float ang = deg * PI_F / 180.0f;
        rel_angles.push_back(ang);
        rel_cos.push_back(std::cos(ang));
        rel_sin.push_back(std::sin(ang));
    }
}

// Final fixed version of the analytic intersection
static float ray_obb_intersect_final(float rox, float roy, float rdx, float rdy,
                                     const Lidar::CachedCar& c, float max_dist) {
    float lox = rox * c.cosH - roy * c.sinH;
    float loy = rox * c.sinH + roy * c.cosH;
    float ldx = rdx * c.cosH - rdy * c.sinH;
    float ldy = rdx * c.sinH + rdy * c.cosH;
    
    float t_min = 0.0f;
    float t_max = max_dist;
    
    auto slab = [&](float o, float d, float h) {
        if (std::abs(d) < 1e-6f) {
            return std::abs(o) <= h;
        }
        float t1 = (-h - o) / d;
        float t2 = (h - o) / d;
        if (t1 > t2) std::swap(t1, t2);
        t_min = std::max(t_min, t1);
        t_max = std::min(t_max, t2);
        return t_min <= t_max;
    };

    if (!slab(lox, ldx, c.hl)) return max_dist;
    if (!slab(loy, ldy, c.hw)) return max_dist;
    
    return (t_min > 0.1f) ? t_min : max_dist;
}

void Lidar::update_from_cache(const Car& self, const std::vector<CachedCar>& obstacles,
                   const RoadGeometry& geom, int width, int height) {
    if ((int)distances.size() != rays) distances.assign(rays, max_dist);

    const float cx = self.state.x;
    const float cy = self.state.y;
    const float cosH = self.cached_cosH;
    const float sinH = self.cached_sinH;

    for (int i = 0; i < rays; ++i) {
        const float dx = cosH * rel_cos[i] - sinH * rel_sin[i];
        const float dy = -(sinH * rel_cos[i] + cosH * rel_sin[i]);

        float final_dist = max_dist;
        for (const auto& c : obstacles) {
            // Skip self based on position proximity (common for global caches)
            if (std::abs(c.x - cx) < 1e-3f && std::abs(c.y - cy) < 1e-3f) continue;

            float rox = cx - c.x;
            float roy = cy - c.y;
            float cross = rox * dy - roy * dx;
            if (cross * cross > c.radius_sq) continue;

            float t = ray_obb_intersect_final(rox, roy, dx, dy, c, max_dist);
            if (t < final_dist) final_dist = t;
        }

        for (float dist = 0.0f; dist < final_dist; dist += step_size) {
            float xf = cx + dx * dist;
            float yf = cy + dy * dist;
            if (xf < 0 || xf >= width || yf < 0 || yf >= height) break;
            if (dist > 0.0f && !geom.is_on_road(xf, yf)) {
                final_dist = dist;
                break;
            }
        }
        distances[i] = final_dist;
    }
}

void Lidar::update_bitmap_from_cache(const Car& self, const std::vector<CachedCar>& obstacles,
                          const BitmapMask& road_mask, int width, int height) {
    if ((int)distances.size() != rays) distances.assign(rays, max_dist);

    const float cx = self.state.x;
    const float cy = self.state.y;
    const float cosH = self.cached_cosH;
    const float sinH = self.cached_sinH;

    for (int i = 0; i < rays; ++i) {
        const float dx = cosH * rel_cos[i] - sinH * rel_sin[i];
        const float dy = -(sinH * rel_cos[i] + cosH * rel_sin[i]);

        float final_dist = max_dist;
        for (const auto& c : obstacles) {
            if (std::abs(c.x - cx) < 1e-3f && std::abs(c.y - cy) < 1e-3f) continue;

            float rox = cx - c.x;
            float roy = cy - c.y;
            float cross = rox * dy - roy * dx;
            if (cross * cross > c.radius_sq) continue;

            float t = ray_obb_intersect_final(rox, roy, dx, dy, c, max_dist);
            if (t < final_dist) final_dist = t;
        }

        float curr_dist = 0.0f;
        while (curr_dist < final_dist) {
            int ix = int(cx + dx * curr_dist);
            int iy = int(cy + dy * curr_dist);
            if (ix < 0 || ix >= width || iy < 0 || iy >= height) break;
            float static_dist = road_mask.get_dist(ix, iy);
            if (curr_dist > 0.0f && static_dist < 1.0f) {
                final_dist = curr_dist;
                break;
            }
            float jump = std::max(step_size, static_dist);
            jump = std::min(jump, 12.0f); 
            curr_dist += jump;
        }
        distances[i] = final_dist;
    }
}

void Lidar::fill_cache(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2) const {
    cache_.clear();
    cache_.reserve(cars1.size() + cars2.size());
    auto build_cache = [&](const std::vector<Car>& list) {
        for (const auto& c : list) {
            if (&c == &self) continue;
            if (std::abs(c.state.x - self.state.x) < 1e-3f && std::abs(c.state.y - self.state.y) < 1e-3f) continue;
            float hl = c.length * 0.5f;
            float hw = c.width * 0.5f;
            cache_.push_back({
                c.state.x, c.state.y,
                c.cached_cosH, c.cached_sinH,
                hl, hw,
                (hl * hl + hw * hw) * 1.1f
            });
        }
    };
    build_cache(cars1);
    build_cache(cars2);
}

void Lidar::update(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2,
                   const RoadGeometry& geom, int width, int height) {
    fill_cache(self, cars1, cars2);
    update_from_cache(self, cache_, geom, width, height);
}

void Lidar::update_bitmap(const Car& self, const std::vector<Car>& cars1, const std::vector<Car>& cars2,
                          const BitmapMask& road_mask, int width, int height) {
    fill_cache(self, cars1, cars2);
    update_bitmap_from_cache(self, cache_, road_mask, width, height);
}

void Lidar::normalized_into(float* out, int out_len) const {
    if (!out || out_len <= 0) return;
    const int n = std::min(out_len, (int)distances.size());
    const float inv = (max_dist > 0.0f) ? (1.0f / max_dist) : 0.0f;
    for (int i = 0; i < n; ++i) out[i] = distances[(size_t)i] * inv;
}

std::vector<float> Lidar::normalized() const {
    std::vector<float> out(distances.size());
    if (!out.empty()) normalized_into(out.data(), (int)out.size());
    return out;
}
