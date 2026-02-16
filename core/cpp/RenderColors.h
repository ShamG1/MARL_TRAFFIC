#pragma once

struct RenderColor {
    float r;
    float g;
    float b;
    float a;
};
// 背景草颜色 34 139 34 为绿色
// 背景颜色 255 255 255 为白色
// 道路颜色 60 60 60 为灰色
// 草地颜色 34 139 34 为绿色
// 中心线颜色 1.0 0.8 0.0 为黄色
// 标记颜色 0.94 0.94 0.94 为白色
// 路线颜色 0.0 1.0 1.0 为青色
// 目标颜色 1.0 0.0 0.0 为红色
// 交通体颜色 150 150 150 为灰色
// 交通头颜色 0.0 0.0 0.0 为黑色
// 代理头颜色 200 200 200 为白色
// 激光雷达射线颜色 0.0 1.0 0.0 为绿色
// 激光雷达命中颜色 1.0 0.0 0.0 为红色
// 车道ID颜色 0.0 0.0 200/255.f 为蓝色
// 车道ID输出颜色 200/255.f 0.0 0.0 为红色
// 文本颜色 0xFFFFFF 为白色
// 道路边界颜色 0.0 0.0 0.0 为黑色
namespace RenderColors {
    inline constexpr RenderColor Background{34/255.f, 139/255.f, 34/255.f, 1.0f};

    inline constexpr RenderColor RoadSurface{60/255.f, 60/255.f, 60/255.f, 1.0f};
    inline constexpr RenderColor Grass{34/255.f, 139/255.f, 34/255.f, 1.0f};

    inline constexpr RenderColor CenterLineYellow{1.0f, 0.8f, 0.0f, 1.0f};
    inline constexpr RenderColor MarkingWhite{0.94f, 0.94f, 0.94f, 1.0f};

    inline constexpr RenderColor RouteCyan{0.0f, 1.0f, 1.0f, 0.8f};
    inline constexpr RenderColor TargetRed{1.0f, 0.0f, 0.0f, 1.0f};

    inline constexpr RenderColor TrafficBodyGray{150/255.f, 150/255.f, 150/255.f, 1.0f};
    inline constexpr RenderColor TrafficHeadBlack{0.0f, 0.0f, 0.0f, 1.0f};

    inline constexpr RenderColor AgentHeadMarker{200/255.f, 200/255.f, 200/255.f, 1.0f};

    inline constexpr RenderColor LidarRayGreen{0.0f, 1.0f, 0.0f, 0.35f};
    inline constexpr RenderColor LidarHitRed{1.0f, 0.0f, 0.0f, 1.0f};

#ifndef _WIN32
    inline constexpr RenderColor LaneIdIn{0.0f, 0.0f, 200/255.f, 1.0f};
    inline constexpr RenderColor LaneIdOut{200/255.f, 0.0f, 0.0f, 1.0f};
#else
    inline constexpr unsigned int LaneIdInRGB = 0x0000C8;
    inline constexpr unsigned int LaneIdOutRGB = 0xC80000;
#endif

    inline constexpr unsigned int HudTextRGB = 0xFFFFFF;

    inline constexpr RenderColor RoadBoundary{0.0f, 0.0f, 0.0f, 1.0f};
}

