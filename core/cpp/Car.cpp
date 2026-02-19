#include "Car.h"
#include <array>
#include <algorithm>
#include <limits>
#include <cmath>

static constexpr float PI_F = 3.14159265358979323846f;

void Car::refresh_pose_cache() {
    cached_cosH = std::cos(state.heading);
    cached_sinH = std::sin(state.heading);
    corners_dirty = true;
}

void Car::update(float throttle, float steer_input, float dt) {
    // Match Scenario.agent.Car.update
    // 1) map inputs
    acc = throttle * MAX_ACC;

    float target_steering = steer_input * MAX_STEERING_ANGLE;
    steering_angle += (target_steering - steering_angle) * 0.2f;

    if (throttle == 0.0f) {
        state.v *= 0.95f;
    }

    // 2) speed update (speed is px/frame, but acc is px/s^2; dt is 1/60)
    state.v += acc * dt;
    if (state.v < 0.0f) state.v = 0.0f;
    if (state.v > PHYSICS_MAX_SPEED) state.v = PHYSICS_MAX_SPEED;

    // heading update (bicycle model)
    if (std::fabs(state.v) > 0.1f) {
        float ang_vel = (state.v / WHEELBASE) * std::tan(steering_angle);
        state.heading += ang_vel;
    }

    // wrap [-pi,pi]
    state.heading = std::fmod(state.heading + PI_F, 2.0f * PI_F);
    if (state.heading < 0) state.heading += 2.0f * PI_F;
    state.heading -= PI_F;

    // Refresh cached trig after heading update
    refresh_pose_cache();

    // 3) position update (NO dt in python)
    state.x += state.v * cached_cosH;
    state.y -= state.v * cached_sinH;
}

void Car::set_path(std::vector<std::pair<float, float>> p) {
    path = std::move(p);
    path_index = 0;
}

void Car::update_path_index() {
    if (path.empty()) {
        path_index = 0;
        return;
    }

    const int n = (int)path.size();
    float min_d = std::numeric_limits<float>::infinity();
    int best_i = path_index;

    // 1) Search in a window around current index (including backwards)
    const int window = 80;
    int start_i = std::max(0, path_index - 20);
    int end_i = std::min(n, path_index + window);

    for (int i = start_i; i < end_i; ++i) {
        const float dx = path[i].first - state.x;
        const float dy = path[i].second - state.y;
        const float d = dx * dx + dy * dy;
        if (d < min_d) {
            min_d = d;
            best_i = i;
        }
    }

    // 2) Global re-localization if we are far from the path
    // 50.0 px is roughly 4-5 car widths, if we are this far, we might have skipped a curve
    if (min_d > 2500.0f) { // 50^2
        for (int i = 0; i < n; ++i) {
            const float dx = path[i].first - state.x;
            const float dy = path[i].second - state.y;
            const float d = dx * dx + dy * dy;
            if (d < min_d) {
                min_d = d;
                best_i = i;
            }
        }
    }

    path_index = best_i;
}

void Car::respawn() {
    state = spawn_state;
    alive = true;
    path_index = 0;
    prev_dist_to_goal = 0.0f;
    prev_action = {0.0f, 0.0f};
    acc = 0.0f;
    steering_angle = 0.0f;

    refresh_pose_cache();
}

std::array<std::pair<float, float>, 4> Car::corners() const {
    if (!corners_dirty) return cached_corners;

    const float hx = width * 0.5f;
    const float hy = length * 0.5f;

    const float cosA = cached_cosH;
    const float sinA = cached_sinH;

    // IMPORTANT: Keep transform consistent with Car::update() which uses:
    //   x += v*cos(h)
    //   y -= v*sin(h)
    // This corresponds to screen coordinates where +y is down, so the rotation
    // for local->world must flip the sign on the sin terms for y.
    auto world = [&](float lx, float ly) {
        float wx = state.x + lx * cosA + ly * sinA;
        float wy = state.y - lx * sinA + ly * cosA;
        return std::make_pair(wx, wy);
    };

    cached_corners = { world( hy,  hx),
             world( hy, -hx),
             world(-hy, -hx),
             world(-hy,  hx) };
    corners_dirty = false;
    return cached_corners;
}

static std::pair<float,float> project(const std::array<std::pair<float,float>,4>& pts,
                                      float ax, float ay) {
    float minP = std::numeric_limits<float>::infinity();
    float maxP = -std::numeric_limits<float>::infinity();
    for (auto [px,py] : pts) {
        float proj = px*ax + py*ay;
        minP = std::min(minP, proj);
        maxP = std::max(maxP, proj);
    }
    return {minP,maxP};
}

bool Car::check_collision(const Car &other) const {
    const auto c1 = corners();
    const auto c2 = other.corners();

    const float ax1 =  cached_cosH;
    const float ay1 =  cached_sinH;
    const float ax2 = -ay1;
    const float ay2 =  ax1;

    const float bx1 =  other.cached_cosH;
    const float by1 =  other.cached_sinH;
    const float bx2 = -by1;
    const float by2 =  bx1;

    std::array<std::pair<float,float>,4> axes = {{
        {ax1, ay1}, {ax2, ay2}, {bx1, by1}, {bx2, by2}
    }};

    for (auto [ax, ay] : axes) {
        auto [min1,max1] = project(c1, ax, ay);
        auto [min2,max2] = project(c2, ax, ay);
        if (max1 < min2 || max2 < min1) return false;
    }
    return true;
}
