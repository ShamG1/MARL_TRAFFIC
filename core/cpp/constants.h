#pragma once

// Mirrored from Scenario/config.py
constexpr int WIDTH  = 1000;                 // 83.3m (at 12px/m)
constexpr int HEIGHT = 1000;                 // 83.3m (at 12px/m)

constexpr float SCALE = 12.0f;              // 12 pixels per meter
constexpr float FPS = 60.0f;                // 60 frames per second
constexpr float DT_DEFAULT = 1.0f / 60.0f;

constexpr float CAR_LENGTH = 54.0f;         // 4.5m
constexpr float CAR_WIDTH = 24.0f;          // 2.0m
constexpr float WHEELBASE = CAR_LENGTH;

constexpr float LANE_WIDTH_PX = 42.0f;      // 3.5m
constexpr float CORNER_RADIUS = 84.0f;      // 7.0m

constexpr float MAX_ACC = 15.0f;            // 1.25 m/s^2 (at 12px/m and dt=1/60)
constexpr float MAX_STEERING_ANGLE = 0.6108652381980153f; // 35 degrees
constexpr float PHYSICS_MAX_SPEED = 4.0f;  // 4 px/frame = 240 px/s = 20.0 m/s (72 km/h) 
