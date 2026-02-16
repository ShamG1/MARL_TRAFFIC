#pragma once

#include <memory>
#include <string>

struct GLFWwindow;

class ImGuiOverlay {
public:
    ImGuiOverlay();
    ~ImGuiOverlay();

    ImGuiOverlay(const ImGuiOverlay&) = delete;
    ImGuiOverlay& operator=(const ImGuiOverlay&) = delete;

    bool init(GLFWwindow* window);
    void shutdown();

    void new_frame();
    void render();

    // Draw text in framebuffer pixel coordinates (origin at top-left)
    void add_text(float x_fb, float y_fb, const std::string& text, unsigned int rgba = 0xFFFFFFFF);

private:
    bool initialized{false};
};
