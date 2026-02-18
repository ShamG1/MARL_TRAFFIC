#pragma once

struct RenderColor {
    float r;
    float g;
    float b;
    float a;
};

namespace RenderColors {
    // Modern dark theme (tuned for sRGB framebuffers)
    inline constexpr RenderColor Background{0.12f, 0.12f, 0.14f, 1.0f};

    inline constexpr RenderColor RoadSurface{0.18f, 0.18f, 0.20f, 1.0f};
    inline constexpr RenderColor Grass{0.10f, 0.15f, 0.10f, 1.0f};

    inline constexpr RenderColor CenterLineYellow{0.95f, 0.75f, 0.0f, 1.0f};
    inline constexpr RenderColor MarkingWhite{0.85f, 0.88f, 0.90f, 0.80f};

    inline constexpr RenderColor RouteCyan{0.0f, 0.90f, 1.0f, 0.60f};
    inline constexpr RenderColor TargetRed{1.0f, 0.20f, 0.20f, 1.0f};

    inline constexpr RenderColor TrafficBodyGray{0.45f, 0.45f, 0.48f, 1.0f};
    inline constexpr RenderColor TrafficHeadBlack{0.05f, 0.05f, 0.05f, 1.0f};

    inline constexpr RenderColor AgentHeadMarker{1.0f, 1.0f, 1.0f, 1.0f};

    inline constexpr RenderColor LidarRayGreen{0.0f, 1.0f, 0.40f, 0.20f};
    inline constexpr RenderColor LidarHitRed{1.0f, 0.10f, 0.30f, 1.0f};

#ifndef _WIN32
    inline constexpr RenderColor LaneIdIn{0.20f, 0.40f, 1.0f, 0.90f};
    inline constexpr RenderColor LaneIdOut{1.0f, 0.30f, 0.30f, 0.90f};
#else
    inline constexpr unsigned int LaneIdInRGB = 0x3366FF;
    inline constexpr unsigned int LaneIdOutRGB = 0xFF4D4D;
#endif

    inline constexpr unsigned int HudTextRGB = 0xE0E0E0;

    inline constexpr RenderColor RoadBoundary{0.05f, 0.05f, 0.05f, 1.0f};
}
