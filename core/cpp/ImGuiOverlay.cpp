#include "ImGuiOverlay.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"

#include <GLFW/glfw3.h>

ImGuiOverlay::ImGuiOverlay() = default;

ImGuiOverlay::~ImGuiOverlay() {
    shutdown();
}

bool ImGuiOverlay::init(GLFWwindow* window) {
    if (initialized) return true;
    if (!window) return false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    // We use OpenGL2 backend (matches existing immediate-mode renderer)
    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
        ImGui::DestroyContext();
        return false;
    }
    if (!ImGui_ImplOpenGL2_Init()) {
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        return false;
    }

    // Slightly nicer font by default
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault();

    initialized = true;
    return true;
}

void ImGuiOverlay::shutdown() {
    if (!initialized) return;
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    initialized = false;
}

void ImGuiOverlay::new_frame() {
    if (!initialized) return;
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiOverlay::render() {
    if (!initialized) return;
    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiOverlay::add_text(float x_fb, float y_fb, const std::string& text, unsigned int rgba) {
    if (!initialized) return;

    const float a = float((rgba >> 24) & 0xFFu) / 255.0f;
    const float b = float((rgba >> 16) & 0xFFu) / 255.0f;
    const float g = float((rgba >> 8) & 0xFFu) / 255.0f;
    const float r = float((rgba >> 0) & 0xFFu) / 255.0f;

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddText(ImVec2(x_fb, y_fb), IM_COL32(int(r * 255), int(g * 255), int(b * 255), int(a * 255)), text.c_str());
}
