#pragma once

#include <vector>
#include <memory>
#include <string>
#include "constants.h"

struct GLFWwindow;

class ImGuiOverlay;

// Forward declaration
class ScenarioEnv;

class Renderer {
public:
    enum ViewMode {
        VIEW_2D = 0,
        VIEW_3D_FOLLOW = 1,
        VIEW_3D_TOP = 2,
    };

    Renderer();
    ~Renderer();

    void set_view_mode(int mode);
    int get_view_mode() const;

    // Disable copy
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Render entire scene based on env state
    void render(const ScenarioEnv& env,
                bool show_lane_ids = false,
                bool show_lidar = false,
                bool show_connections = false);

    // Input / window state (GLFW)
    bool window_should_close() const;
    void poll_events() const;
    bool key_pressed(int glfw_key) const;

    // Whether initialization succeeded
    bool ok() const { return initialized; }
private:
    bool init_glfw();
    void draw_bitmap_background(const ScenarioEnv& env) const;
    void draw_cars(const ScenarioEnv& env) const;
    void draw_lidar(const ScenarioEnv& env) const;
    void draw_lane_ids(const ScenarioEnv& env) const;
    void draw_hud(const ScenarioEnv& env) const;
    void draw_route(const ScenarioEnv& env) const;
    void draw_connections(const ScenarioEnv& env) const;
    void update_view_box(const ScenarioEnv& env);

#ifdef _WIN32
    // Win32 GDI overlay text (better HUD font)
    void gdi_begin_frame(int fb_w, int fb_h) const;
    void gdi_draw_text_px(int x, int y, const std::string& text, unsigned int rgb) const;
    void gdi_end_frame() const;
#endif

    bool initialized{false};
    struct Impl;
    std::unique_ptr<Impl> impl;

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    std::unique_ptr<ImGuiOverlay> imgui;
};
