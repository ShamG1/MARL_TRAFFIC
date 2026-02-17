#include "Renderer.h"
#include "ScenarioEnv.h"
#include "RenderColors.h"
#include "ImGuiOverlay.h"

#ifndef _WIN32
#include "imgui.h"
#endif

#include <GLFW/glfw3.h>

#ifndef _WIN32
#include "stb_image.h"
#endif

#ifdef _WIN32
#define NOMINMAX
#define GLFW_EXPOSE_NATIVE_WIN32
#include <Windows.h>
#include <GLFW/glfw3native.h>
#endif

#include <cmath>
#include <iostream>
#include <array>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cstring>

#include <fstream>
#include <sstream>

struct ObjMesh {
    struct Vertex { float x, y, z; };
    struct Face { int v[3]; };
    struct Material {
        float r, g, b;
        std::string name;
    };
    
    std::vector<Vertex> vertices;
    std::vector<Vertex> vert_normals; 
    std::map<std::string, std::vector<Face>> material_groups;
    std::map<std::string, Material> materials;
    
    float center_x{0.0f}, center_y{0.0f}, center_z{0.0f};
    float scale{1.0f};

    bool load_mtl(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line, cur_mtl;
        while (std::getline(f, line)) {
            std::istringstream s(line);
            std::string cmd; s >> cmd;
            if (cmd == "newmtl") {
                s >> cur_mtl;
                materials[cur_mtl] = {0.8f, 0.8f, 0.8f, cur_mtl};
            } else if (cmd == "Kd" && !cur_mtl.empty()) {
                s >> materials[cur_mtl].r >> materials[cur_mtl].g >> materials[cur_mtl].b;
            }
        }
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        
        vertices.clear(); material_groups.clear(); vert_normals.clear(); materials.clear();
        
        std::string mtl_lib;
        std::string cur_mtl = "default";
        std::string line;
        while (std::getline(f, line)) {
            if (line.substr(0, 7) == "mtllib ") {
                mtl_lib = line.substr(7);
                // Try to load mtl from same directory
                size_t last_slash = path.find_last_of("/\\");
                std::string dir = (last_slash != std::string::npos) ? path.substr(0, last_slash + 1) : "";
                load_mtl(dir + mtl_lib);
            } else if (line.substr(0, 7) == "usemtl ") {
                cur_mtl = line.substr(7);
                // Ensure the string is trimmed
                cur_mtl.erase(cur_mtl.find_last_not_of(" \n\r\t") + 1);
            } else if (line.substr(0, 2) == "v ") {
                std::istringstream s(line.substr(2));
                Vertex v; s >> v.x >> v.y >> v.z;
                vertices.push_back(v);
            } else if (line.substr(0, 2) == "f ") {
                std::istringstream s(line.substr(2));
                std::vector<int> face_indices;
                std::string v_str;
                while (s >> v_str) {
                    try {
                        size_t first_slash = v_str.find('/');
                        int idx = std::stoi(v_str.substr(0, first_slash));
                        face_indices.push_back(idx > 0 ? idx - 1 : (int)vertices.size() + idx);
                    } catch (...) {}
                }
                for (size_t i = 1; i + 1 < face_indices.size(); ++i) {
                    material_groups[cur_mtl].push_back({{face_indices[0], face_indices[i], face_indices[i+1]}});
                }
            }
        }
        if (vertices.empty()) return false;

        // Compute BBox and Center
        float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9, min_z = 1e9, max_z = -1e9;
        for (const auto& v : vertices) {
            min_x = std::min(min_x, v.x); max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y); max_y = std::max(max_y, v.y);
            min_z = std::min(min_z, v.z); max_z = std::max(max_z, v.z);
        }
        center_x = (min_x + max_x) * 0.5f;
        center_y = min_y; 
        center_z = (min_z + max_z) * 0.5f;

        float size_z = max_z - min_z;
        scale = CAR_LENGTH / (size_z > 0.1f ? size_z : 1.0f);

        // Compute Smooth Vert Normals
        vert_normals.assign(vertices.size(), {0, 0, 0});
        for (const auto& pair : material_groups) {
            for (const auto& f : pair.second) {
                const auto& v0 = vertices[f.v[0]], v1 = vertices[f.v[1]], v2 = vertices[f.v[2]];
                float ax = v1.x - v0.x, ay = v1.y - v0.y, az = v1.z - v0.z;
                float bx = v2.x - v0.x, by = v2.y - v0.y, bz = v2.z - v0.z;
                float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
                for (int i = 0; i < 3; ++i) {
                    vert_normals[f.v[i]].x += nx; vert_normals[f.v[i]].y += ny; vert_normals[f.v[i]].z += nz;
                }
            }
        }
        for (auto& n : vert_normals) {
            float len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
            if (len > 1e-6f) { n.x /= len; n.y /= len; n.z /= len; }
        }
        return true;
    }

    void draw(float override_r = -1.0f, float override_g = -1.0f, float override_b = -1.0f) const {
        for (const auto& pair : material_groups) {
            const std::string& mtl_name = pair.first;
            const auto& faces_vec = pair.second;

            // Simple logic: if material name contains "body" or "paint", use override color
            bool is_body = false;
            std::string lower_name = mtl_name;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
            if (lower_name.find("body") != std::string::npos || 
                lower_name.find("paint") != std::string::npos || 
                lower_name.find("car_color") != std::string::npos ||
                lower_name == "blue" ||
                lower_name == "lightblue" ||
                lower_name.rfind("material.", 0) == 0) {
                is_body = true;
            }

            if (is_body && override_r >= 0) {
                glColor3f(override_r, override_g, override_b);
            } else if (materials.count(mtl_name)) {
                const auto& m = materials.at(mtl_name);
                glColor3f(m.r, m.g, m.b);
            } else {
                // Fallback palette when .mtl is missing
                if (lower_name == "black") {
                    glColor3f(0.05f, 0.05f, 0.05f);
                } else if (lower_name == "grey" || lower_name == "gray") {
                    glColor3f(0.45f, 0.45f, 0.48f);
                } else if (lower_name == "windows" || lower_name == "window" || lower_name == "glass") {
                    glColor3f(0.25f, 0.32f, 0.38f);
                } else if (lower_name == "headlights" || lower_name == "headlight") {
                    glColor3f(0.95f, 0.95f, 0.85f);
                } else if (lower_name == "taillights" || lower_name == "taillight") {
                    glColor3f(0.85f, 0.12f, 0.12f);
                } else {
                    glColor3f(0.8f, 0.8f, 0.8f);
                }
            }

            glBegin(GL_TRIANGLES);
            for (const auto& f : faces_vec) {
                for (int i = 0; i < 3; ++i) {
                    const auto& n = vert_normals[f.v[i]];
                    glNormal3f(n.x, n.y, n.z);
                    const auto& v = vertices[f.v[i]];
                    glVertex3f((v.x - center_x), (v.y - center_y), (v.z - center_z));
                }
            }
            glEnd();
        }
    }
};

struct Renderer::Impl {
    GLFWwindow* window{nullptr};
    ObjMesh mesh_car;
    bool meshes_loaded{false};


    // Mouse input
    double mouse_x{0.0};
    double mouse_y{0.0};
    double last_mouse_x{0.0};
    double last_mouse_y{0.0};
    bool mouse_left_down{false};
    bool mouse_right_down{false};
    double scroll_y{0.0};

    // Camera params (3D_TOP)
    float top_yaw{0.0f};
    float top_pitch{0.70f};
    float top_dist{900.0f};
    float top_center_x{WIDTH * 0.5f};
    float top_center_z{HEIGHT * 0.5f};

    // Camera params (3D_FOLLOW)
    float follow_yaw{0.0f};
    float follow_pitch{0.5f};
    float follow_dist{120.0f};

#ifdef _WIN32
    HWND hwnd{nullptr};
    HDC hdc{nullptr};
    HFONT font{nullptr};
    HFONT font_lane_ids{nullptr};
    HICON hicon_small{nullptr};
    HICON hicon_big{nullptr};
#endif
    int fb_w{0};
    int fb_h{0};

    // Dynamic view-box (logical pixels)
    float v_min_x{0.0f};
    float v_max_x{(float)WIDTH};
    float v_min_y{0.0f};
    float v_max_y{(float)HEIGHT};

    int view_mode{0}; // 0=2D, 1=3D

    // Bitmap background texture cache
    bool bg_tex_valid{false};
    GLuint bg_tex{0};
    int bg_w{0};
    int bg_h{0};
};

// Helper for 3D perspective matrix (equivalent to gluPerspective)
static void setup_perspective(float fov_deg, float aspect, float znear, float zfar) {
    float f = 1.0f / std::tan(fov_deg * 3.14159265f / 360.0f);
    float m[16] = {
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (zfar + znear) / (znear - zfar), -1,
        0, 0, (2.0f * zfar * znear) / (znear - zfar), 0
    };
    glLoadMatrixf(m);
}

// Simple LookAt implementation
static void setup_lookat(float eyex, float eyey, float eyez,
                         float centerx, float centery, float centerz,
                         float upx, float upy, float upz) {
    float f[3] = { centerx - eyex, centery - eyey, centerz - eyez };
    float len = std::sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    f[0] /= len; f[1] /= len; f[2] /= len;

    float up[3] = { upx, upy, upz };
    float s[3] = { f[1]*up[2] - f[2]*up[1], f[2]*up[0] - f[0]*up[2], f[0]*up[1] - f[1]*up[0] };
    len = std::sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    s[0] /= len; s[1] /= len; s[2] /= len;

    float u[3] = { s[1]*f[2] - s[2]*f[1], s[2]*f[0] - s[0]*f[2], s[0]*f[1] - s[1]*f[0] };

    float m[16] = {
        s[0], u[0], -f[0], 0,
        s[1], u[1], -f[1], 0,
        s[2], u[2], -f[2], 0,
        0, 0, 0, 1
    };
    glMultMatrixf(m);
    glTranslatef(-eyex, -eyey, -eyez);
}

// Helper for dynamic NDC mapping based on a view box (used for 2D mode)
static inline float ndc_x(float px, float min_x, float max_x) {
    float range = max_x - min_x;
    if (range < 1e-3f) range = 1.0f;
    return (px - min_x) / range * 2.0f - 1.0f;
}

static inline float ndc_y(float py, float min_y, float max_y) {
    float range = max_y - min_y;
    if (range < 1e-3f) range = 1.0f;
    return 1.0f - (py - min_y) / range * 2.0f;
}

static void draw_rect_ndc(float x_px, float y_px, float w_px, float h_px, float r, float g, float b, float a, 
                          float min_x, float max_x, float min_y, float max_y) {
    float x0 = ndc_x(x_px, min_x, max_x);
    float y0 = ndc_y(y_px, min_y, max_y);
    float x1 = ndc_x(x_px + w_px, min_x, max_x);
    float y1 = ndc_y(y_px + h_px, min_y, max_y);
    glColor4f(r, g, b, a);
    glBegin(GL_QUADS);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();
}

static void draw_line_px(float x0, float y0, float x1, float y1, float width, float r, float g, float b, float a,
                           float min_x, float max_x, float min_y, float max_y) {
    glLineWidth(width);
    glColor4f(r, g, b, a);
    glBegin(GL_LINES);
    glVertex2f(ndc_x(x0, min_x, max_x), ndc_y(y0, min_y, max_y));
    glVertex2f(ndc_x(x1, min_x, max_x), ndc_y(y1, min_y, max_y));
    glEnd();
}

static void draw_circle_px(float cx, float cy, float radius, int segments, float r, float g, float b,
                            float min_x, float max_x, float min_y, float max_y) {
    glColor3f(r, g, b);
    glBegin(GL_TRIANGLE_FAN);
    for (int i = 0; i <= segments; i++) {
        constexpr float PI_F = 3.14159265358979323846f;
        float a = 2.0f * PI_F * float(i) / float(segments);
        float x = cx + std::cos(a) * radius;
        float y = cy + std::sin(a) * radius;
        glVertex2f(ndc_x(x, min_x, max_x), ndc_y(y, min_y, max_y));
    }
    glEnd();
}

Renderer::Renderer() {
    if(!init_glfw()) return;
    impl = std::make_unique<Impl>();
    imgui = std::make_unique<ImGuiOverlay>();

    // Load OBJ models
    std::string assets_dir = std::string(CPP_ASSETS_DIR) + "/";
#ifdef _WIN32
    std::replace(assets_dir.begin(), assets_dir.end(), '/', '\\');
#endif
    if (impl->mesh_car.load(assets_dir + "CAR.obj")) {
        impl->meshes_loaded = true;
    } else {
        std::cerr << "[Renderer] Warning: Failed to load car OBJ models from " << assets_dir << std::endl;
    }

    // We use immediate-mode OpenGL (glBegin/glEnd). Core profile removes these APIs,
    // so we must request a COMPAT profile.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    impl->window = glfwCreateWindow(WIDTH*1.5, HEIGHT*1.5, "DriveSimX", nullptr, nullptr);
    if(!impl->window){
        glfwTerminate();
        return;
    }
    // Lock window aspect ratio to 1:1 and force initial square size
    glfwSetWindowAspectRatio(impl->window, 1, 1);
    glfwSetWindowSize(impl->window, WIDTH, HEIGHT);

#ifdef _WIN32
    // Set window icon (ICO only; PNG via GDI+ was unstable)
    impl->hwnd = glfwGetWin32Window(impl->window);
    if(impl->hwnd){
        const std::wstring ico_path = to_wide(std::string(CPP_ASSETS_DIR) + "\\icon.ico");

        HICON icon_big = static_cast<HICON>(
            LoadImageW(nullptr, ico_path.c_str(), IMAGE_ICON, 0, 0,
                       LR_LOADFROMFILE | LR_DEFAULTSIZE)
        );
        HICON icon_small = static_cast<HICON>(
            LoadImageW(nullptr, ico_path.c_str(), IMAGE_ICON, 16, 16,
                       LR_LOADFROMFILE)
        );
        if(icon_big){
            SendMessageW(impl->hwnd, WM_SETICON, ICON_BIG, (LPARAM)icon_big);
            impl->hicon_big = icon_big;
        }
        if(icon_small){
            SendMessageW(impl->hwnd, WM_SETICON, ICON_SMALL, (LPARAM)icon_small);
            impl->hicon_small = icon_small;
        }
    }
#else
    // Linux/macOS: set icon via GLFW (expects RGBA pixels)
    const std::string png_path = std::string(CPP_ASSETS_DIR) + "/icon.png";
    int icon_w = 0, icon_h = 0, icon_comp = 0;
    unsigned char* pixels = stbi_load(png_path.c_str(), &icon_w, &icon_h, &icon_comp, 4);
    if(pixels && icon_w > 0 && icon_h > 0){
        GLFWimage image;
        image.width = icon_w;
        image.height = icon_h;
        image.pixels = pixels;
        glfwSetWindowIcon(impl->window, 1, &image);
    }
    if(pixels) stbi_image_free(pixels);
#endif

    glfwMakeContextCurrent(impl->window);

    glfwSetWindowUserPointer(impl->window, this);
    glfwSetScrollCallback(impl->window, &Renderer::scroll_callback);
    glfwSetCursorPosCallback(impl->window, &Renderer::cursor_pos_callback);
    glfwSetMouseButtonCallback(impl->window, &Renderer::mouse_button_callback);
    glfwSwapInterval(1);

    if (imgui) {
        imgui->init(impl->window);
    }
    // No GLAD in this build: using immediate-mode OpenGL functions provided via system GL headers.
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

#ifdef _WIN32
    impl->hwnd = glfwGetWin32Window(impl->window);
    impl->hdc = nullptr;
    impl->font = CreateFontW(
        -18, 0, 0, 0,
        FW_NORMAL,
        FALSE, FALSE, FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"Segoe UI");

    // Slightly smaller font for lane IDs (independent from HUD)
    impl->font_lane_ids = CreateFontW(
        -12, 0, 0, 0,
        FW_NORMAL,
        FALSE, FALSE, FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"Segoe UI");
#endif

    initialized = true;
}

Renderer::~Renderer(){
    if (imgui) {
        imgui->shutdown();
    }

    if (impl) {
        if (impl->bg_tex_valid && impl->bg_tex != 0) {
            glDeleteTextures(1, &impl->bg_tex);
            impl->bg_tex = 0;
            impl->bg_tex_valid = false;
        }
    }

#ifdef _WIN32
    if(impl){
        if(impl->hdc){
            ReleaseDC(impl->hwnd, impl->hdc);
            impl->hdc = nullptr;
        }
        if(impl->font){
            DeleteObject(impl->font);
            impl->font = nullptr;
        }
        if(impl->font_lane_ids){
            DeleteObject(impl->font_lane_ids);
            impl->font_lane_ids = nullptr;
        }
        if(impl->hicon_big){
            DestroyIcon(impl->hicon_big);
            impl->hicon_big = nullptr;
        }
        if(impl->hicon_small){
            DestroyIcon(impl->hicon_small);
            impl->hicon_small = nullptr;
        }
    }
#endif
    if(impl && impl->window){
        glfwDestroyWindow(impl->window);
        glfwTerminate();
    }
}

bool Renderer::init_glfw(){
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW" << std::endl;
        return false;
    }
    return true;
}

void Renderer::set_view_mode(int mode) {
    if (impl) impl->view_mode = mode;
}

int Renderer::get_view_mode() const {
    return impl ? impl->view_mode : 0;
}

void Renderer::scroll_callback(GLFWwindow* window, double, double yoffset) {
    if (!window) return;
    auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self || !self->impl) return;
    self->impl->scroll_y += yoffset;
}

void Renderer::cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!window) return;
    auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self || !self->impl) return;
    self->impl->mouse_x = xpos;
    self->impl->mouse_y = ypos;
}

void Renderer::mouse_button_callback(GLFWwindow* window, int button, int action, int) {
    if (!window) return;
    auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self || !self->impl) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        self->impl->mouse_left_down = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        self->impl->mouse_right_down = (action == GLFW_PRESS);
    }

    // Reset drag origin on press
    if (action == GLFW_PRESS) {
        self->impl->last_mouse_x = self->impl->mouse_x;
        self->impl->last_mouse_y = self->impl->mouse_y;
    }
}

bool Renderer::window_should_close() const {
    if(!impl || !impl->window) return true;
    return glfwWindowShouldClose(impl->window) != 0;
}

void Renderer::poll_events() const {
    glfwPollEvents();
}

bool Renderer::key_pressed(int glfw_key) const {
    if(!impl || !impl->window) return false;
    return glfwGetKey(impl->window, glfw_key) == GLFW_PRESS;
}

void Renderer::draw_bitmap_background(const ScenarioEnv& env) const {
    if (!impl) return;
    if (!env.use_bitmap_scenario) return;

    const int W = WIDTH;
    const int H = HEIGHT;

    if (!impl->bg_tex_valid || impl->bg_w != W || impl->bg_h != H) {
        if (impl->bg_tex != 0) {
            glDeleteTextures(1, &impl->bg_tex);
            impl->bg_tex = 0;
        }

        std::vector<unsigned char> rgb;
        rgb.resize((size_t)W * (size_t)H * 3);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int idx = (y * W + x) * 3;
                const bool is_road = env.bitmap_road.at(x, y) > 0;
                const bool is_line = env.bitmap_line.at(x, y) > 0;
                const bool is_dash = env.bitmap_dash.at(x, y) > 0;

                if (is_dash) {
                    rgb[idx + 0] = 255;
                    rgb[idx + 1] = 255;
                    rgb[idx + 2] = 255;
                } else if (is_line) {
                    rgb[idx + 0] = (unsigned char)std::clamp(int(RenderColors::CenterLineYellow.r * 255.0f), 0, 255);
                    rgb[idx + 1] = (unsigned char)std::clamp(int(RenderColors::CenterLineYellow.g * 255.0f), 0, 255);
                    rgb[idx + 2] = (unsigned char)std::clamp(int(RenderColors::CenterLineYellow.b * 255.0f), 0, 255);
                } else if (is_road) {
                    rgb[idx + 0] = (unsigned char)std::clamp(int(RenderColors::RoadSurface.r * 255.0f), 0, 255);
                    rgb[idx + 1] = (unsigned char)std::clamp(int(RenderColors::RoadSurface.g * 255.0f), 0, 255);
                    rgb[idx + 2] = (unsigned char)std::clamp(int(RenderColors::RoadSurface.b * 255.0f), 0, 255);
                } else {
                    rgb[idx + 0] = (unsigned char)std::clamp(int(RenderColors::Grass.r * 255.0f), 0, 255);
                    rgb[idx + 1] = (unsigned char)std::clamp(int(RenderColors::Grass.g * 255.0f), 0, 255);
                    rgb[idx + 2] = (unsigned char)std::clamp(int(RenderColors::Grass.b * 255.0f), 0, 255);
                }
            }
        }

        glGenTextures(1, &impl->bg_tex);
        glBindTexture(GL_TEXTURE_2D, impl->bg_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

        // Flip vertically when uploading: OpenGL texture coords origin is bottom-left.
        // We keep source y=0 at top; easiest is to upload rows reversed.
        std::vector<unsigned char> flipped;
        flipped.resize(rgb.size());
        for (int y = 0; y < H; ++y) {
            std::memcpy(&flipped[(size_t)y * (size_t)W * 3],
                        &rgb[(size_t)(H - 1 - y) * (size_t)W * 3],
                        (size_t)W * 3);
        }

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, flipped.data());

        impl->bg_tex_valid = true;
        impl->bg_w = W;
        impl->bg_h = H;
    }

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, impl->bg_tex);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    if (impl->view_mode == VIEW_2D) {
        // Map the view box to the texture coordinates
        float tx0 = impl->v_min_x / (float)WIDTH;
        float ty0 = 1.0f - impl->v_max_y / (float)HEIGHT; // Flip Y for OpenGL
        float tx1 = impl->v_max_x / (float)WIDTH;
        float ty1 = 1.0f - impl->v_min_y / (float)HEIGHT;

        glBegin(GL_QUADS);
        glTexCoord2f(tx0, ty0); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(tx1, ty0); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(tx1, ty1); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(tx0, ty1); glVertex2f(-1.0f,  1.0f);
        glEnd();
    } else {
        // Correct 3D ground: map texture to world coordinates on Y=0 plane
        // Disable lighting and culling for ground to ensure visibility
        glDisable(GL_LIGHTING);
        glDisable(GL_CULL_FACE);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, impl->bg_tex);
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

        // Explicit CCW order for the ground quad
        // Flip texture V coordinate to match the flipped background upload (y=0 at top)
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, -0.5f, 0.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex3f((float)WIDTH, -0.5f, 0.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex3f((float)WIDTH, -0.5f, (float)HEIGHT);
        glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, -0.5f, (float)HEIGHT);
        glEnd();

        glEnable(GL_CULL_FACE);
        glEnable(GL_LIGHTING);
    }

    glDisable(GL_TEXTURE_2D);
}

void Renderer::update_view_box(const ScenarioEnv& env) {
    if (!impl || !env.use_bitmap_scenario) return;

    static bool first_bbox_log = true;
    int min_x = WIDTH, max_x = 0;
    int min_y = HEIGHT, max_y = 0;
    bool found = false;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            if (env.bitmap_road.at(x, y) > 0) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
                found = true;
            }
        }
    }

    if (found) {
        const float padding = 2.0f;
        float rb_min_x = std::max(0.0f, (float)min_x - padding);
        float rb_max_x = std::min((float)WIDTH, (float)max_x + padding);
        float rb_min_y = std::max(0.0f, (float)min_y - padding);
        float rb_max_y = std::min((float)HEIGHT, (float)max_y + padding);

        float w = rb_max_x - rb_min_x;
        float h = rb_max_y - rb_min_y;
        float side = std::max(w, h);
        float cx = (rb_min_x + rb_max_x) * 0.5f;
        float cy = (rb_min_y + rb_max_y) * 0.5f;

        float half_side = side * 0.5f;
        float new_min_x = cx - half_side;
        float new_max_x = cx + half_side;
        float new_min_y = cy - half_side;
        float new_max_y = cy + half_side;

        // Clamp to map bounds
        if (new_min_x < 0.0f) { new_max_x -= new_min_x; new_min_x = 0.0f; }
        if (new_max_x > WIDTH) { new_min_x -= (new_max_x - WIDTH); new_max_x = (float)WIDTH; }
        if (new_min_y < 0.0f) { new_max_y -= new_min_y; new_min_y = 0.0f; }
        if (new_max_y > HEIGHT) { new_min_y -= (new_max_y - HEIGHT); new_max_y = (float)HEIGHT; }

        // Recalculate side after clamp to ensure it stays square if possible within map
        float final_w = new_max_x - new_min_x;
        float final_h = new_max_y - new_min_y;
        float final_side = std::min(final_w, final_h);
        // Re-center slightly if map bounds forced non-squareness
        if (final_w > final_side) {
             float diff = (final_w - final_side) * 0.5f;
             new_min_x += diff; new_max_x -= diff;
        }
        if (final_h > final_side) {
             float diff = (final_h - final_side) * 0.5f;
             new_min_y += diff; new_max_y -= diff;
        }

        // Check if road boundaries significantly changed to trigger a window resize
        bool changed = (std::abs(new_min_x - impl->v_min_x) > 1.0f ||
                        std::abs(new_max_x - impl->v_max_x) > 1.0f ||
                        std::abs(new_min_y - impl->v_min_y) > 1.0f ||
                        std::abs(new_max_y - impl->v_max_y) > 1.0f);

        if (changed) {
            impl->v_min_x = new_min_x;
            impl->v_max_x = new_max_x;
            impl->v_min_y = new_min_y;
            impl->v_max_y = new_max_y;

            // Always log when view-box changes to help debug scenario alignment
            std::cerr << "[Renderer] Scenario: " << env.scenario_name 
                      << " | bbox(px): " << min_x << "," << min_y << " to " << max_x << "," << max_y 
                      << " | view-box: " << impl->v_min_x << "," << impl->v_min_y 
                      << " to " << impl->v_max_x << "," << impl->v_max_y << std::endl;

            // Auto-resize window to a square.
            int target = 900;
            glfwSetWindowSize(impl->window, target, target);
        }
    } else {
        impl->v_min_x = 0.0f;
        impl->v_max_x = (float)WIDTH;
        impl->v_min_y = 0.0f;
        impl->v_max_y = (float)HEIGHT;
    }
}

void Renderer::render(const ScenarioEnv& env, bool show_lane_ids, bool show_lidar){
    if(!initialized) return;
    glfwMakeContextCurrent(impl->window);

    // Update view box to fit road
    update_view_box(env);

    // Square viewport to match square view-box (avoid horizontal/vertical bars)
    int full_w = WIDTH;
    int full_h = HEIGHT;
    glfwGetFramebufferSize(impl->window, &full_w, &full_h);

    static bool first_render_log = false;
    int win_w, win_h;
    glfwGetWindowSize(impl->window, &win_w, &win_h);

    const int view = (full_w < full_h) ? full_w : full_h;
    const int vp_x = (full_w - view) / 2;
    const int vp_y = (full_h - view) / 2;

    if (first_render_log) {
        std::cerr << "[Renderer] Render Info: "
                  << "win(" << win_w << "x" << win_h << ") "
                  << "fb(" << full_w << "x" << full_h << ") "
                  << "vp(x=" << vp_x << ",y=" << vp_y << ",view=" << view << ")\n";
        first_render_log = false;
    }

    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(vp_x, vp_y, view, view);

    glEnable(GL_SCISSOR_TEST);
    glScissor(vp_x, vp_y, view, view);
    glClearColor(RenderColors::Background.r, RenderColors::Background.g, RenderColors::Background.b, RenderColors::Background.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

    if (impl->view_mode == VIEW_2D) {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
    } else {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glEnable(GL_NORMALIZE);

        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

        // Bright / cartoony lighting
        float light_pos[] = { 0.3f, 1.0f, 0.2f, 0.0f }; // directional
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
        float light_amb[] = { 0.55f, 0.55f, 0.60f, 1.0f };
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
        float light_diff[] = { 0.95f, 0.95f, 0.95f, 1.0f };
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diff);
        float light_spec[] = { 0.25f, 0.25f, 0.25f, 1.0f };
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_spec);

        float mat_spec[] = { 0.20f, 0.20f, 0.20f, 1.0f };
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_spec);
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 24.0f);

        glMatrixMode(GL_PROJECTION);
        setup_perspective(45.0f, 1.0f, 1.0f, 5000.0f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // --- Mouse-driven camera controls (3D) ---
        double dx = impl->mouse_x - impl->last_mouse_x;
        double dy = impl->mouse_y - impl->last_mouse_y;
        impl->last_mouse_x = impl->mouse_x;
        impl->last_mouse_y = impl->mouse_y;

        // Zoom via scroll wheel
        if (impl->scroll_y != 0.0) {
            if (impl->view_mode == VIEW_3D_TOP) {
                impl->top_dist = std::clamp(impl->top_dist * (float)std::pow(0.9, impl->scroll_y), 200.0f, 4000.0f);
            } else if (impl->view_mode == VIEW_3D_FOLLOW) {
                impl->follow_dist = std::clamp(impl->follow_dist * (float)std::pow(0.9, impl->scroll_y), 30.0f, 800.0f);
            }
            impl->scroll_y = 0.0;
        }

        if (impl->view_mode == VIEW_3D_TOP) {
            // Left drag: pan
            if (impl->mouse_left_down) {
                float pan_speed = impl->top_dist * 0.0015f;
                impl->top_center_x -= (float)dx * pan_speed;
                impl->top_center_z += (float)dy * pan_speed;
            }
            // Right drag: orbit
            if (impl->mouse_right_down) {
                impl->top_yaw += (float)dx * 0.005f;
                impl->top_pitch = std::clamp(impl->top_pitch + (float)dy * 0.005f, 0.15f, 1.45f);
            }

            float cy = std::cos(impl->top_yaw);
            float sy = std::sin(impl->top_yaw);
            float cp = std::cos(impl->top_pitch);
            float sp = std::sin(impl->top_pitch);

            float ex = impl->top_center_x + impl->top_dist * cp * cy;
            float ey = impl->top_dist * sp;
            float ez = impl->top_center_z + impl->top_dist * cp * sy;
            setup_lookat(ex, ey, ez, impl->top_center_x, 0.0f, impl->top_center_z, 0, 1, 0);
        } else if (impl->view_mode == VIEW_3D_FOLLOW && !env.cars.empty() && env.cars[0].alive) {
            const auto& ego = env.cars[0];

            // Right drag: rotate around ego
            if (impl->mouse_right_down) {
                impl->follow_yaw += (float)dx * 0.005f;
                impl->follow_pitch = std::clamp(impl->follow_pitch + (float)dy * 0.005f, 0.05f, 1.2f);
            }

            float base_yaw = -ego.state.heading;
            float yaw = base_yaw + impl->follow_yaw;
            float cy = std::cos(yaw);
            float sy = std::sin(yaw);
            float cp = std::cos(impl->follow_pitch);
            float sp = std::sin(impl->follow_pitch);

            float ex = ego.state.x - impl->follow_dist * cp * cy;
            float ey = 8.0f + impl->follow_dist * sp;
            float ez = ego.state.y + impl->follow_dist * cp * sy;
            setup_lookat(ex, ey, ez, ego.state.x, 3.0f, ego.state.y, 0, 1, 0);
        } else {
            // Fallback
            setup_lookat(WIDTH / 2.0f, 600.0f, HEIGHT / 2.0f + 300.0f,
                         WIDTH / 2.0f, 0.0f, HEIGHT / 2.0f,
                         0, 1, 0);
        }
    }

    if (env.use_bitmap_scenario) {
        draw_bitmap_background(env);
    } else {
        throw std::runtime_error("Renderer: Bitmap scenario required but not loaded.");
    }

    draw_route(env);
    draw_cars(env);
    if(show_lidar) draw_lidar(env);

#ifdef _WIN32
    if(show_lane_ids){
        gdi_begin_frame(full_w, full_h);
        draw_lane_ids(env);
        gdi_end_frame();
    }
#else
    if (imgui) {
        imgui->new_frame();
    }

    if (show_lane_ids) {
        draw_lane_ids(env);
    }

    draw_hud(env);

    if (imgui) {
        imgui->render();
    }
#endif

    glfwSwapBuffers(impl->window);

    // Keep GLFW input responsive even if Python doesn't call poll_events()
    glfwPollEvents();
}

#ifdef _WIN32

static std::wstring to_wide(const std::string& s){
    if(s.empty()) return std::wstring();
    int needed = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if(needed <= 0){
        std::wstring out;
        out.reserve(s.size());
        for(unsigned char ch : s) out.push_back((wchar_t)ch);
        return out;
    }
    std::wstring w;
    w.resize((size_t)needed - 1);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), needed);
    return w;
}

void Renderer::gdi_begin_frame(int fb_w, int fb_h) const{
    if(!impl || !impl->hwnd) return;
    if(!impl->hdc) impl->hdc = GetDC(impl->hwnd);
    impl->fb_w = fb_w;
    impl->fb_h = fb_h;

    if(!impl->hdc) return;
    SetBkMode(impl->hdc, TRANSPARENT);
    if(impl->font) SelectObject(impl->hdc, impl->font);
}

void Renderer::gdi_draw_text_px(int x, int y, const std::string& text, unsigned int rgb) const{
    if(!impl || !impl->hdc) return;

    const COLORREF color = RGB((rgb >> 16) & 0xFFu, (rgb >> 8) & 0xFFu, rgb & 0xFFu);
    SetTextColor(impl->hdc, color);

    auto w = to_wide(text);
    TextOutW(impl->hdc, x, y, w.c_str(), (int)w.size());
}

void Renderer::gdi_end_frame() const{
    if(!impl || !impl->hwnd || !impl->hdc) return;
    ReleaseDC(impl->hwnd, impl->hdc);
    impl->hdc = nullptr;
}

void Renderer::draw_lane_ids(const ScenarioEnv& env) const{
    if(!impl || !impl->hdc) return;

    HGDIOBJ old_font = nullptr;
    if(impl->font_lane_ids){
        old_font = SelectObject(impl->hdc, impl->font_lane_ids);
    }
    const unsigned int in_color = RenderColors::LaneIdInRGB;
    const unsigned int out_color = RenderColors::LaneIdOutRGB;

    // To match the old layout, we need to know the viewport transform
    const int view = (impl->fb_w < impl->fb_h) ? impl->fb_w : impl->fb_h;
    const int vp_x = (impl->fb_w - view) / 2;
    const int vp_y = (impl->fb_h - view) / 2;

    // Stable draw order to avoid label flicker when texts overlap
    std::vector<std::pair<std::string, std::pair<float,float>>> items;
    items.reserve(env.lane_layout.points.size());
    for(const auto& kv : env.lane_layout.points){
        items.push_back(kv);
    }

    auto key = [](const std::string& id){
        int group = 2; // others
        if(id.rfind("IN_", 0) == 0) group = 0;
        else if(id.rfind("OUT_", 0) == 0) group = 1;

        int num = -1;
        size_t us = id.find('_');
        if(us != std::string::npos){
            try { num = std::stoi(id.substr(us + 1)); } catch(...) { num = -1; }
        }
        return std::pair<int,int>(group, num);
    };

    std::sort(items.begin(), items.end(), [&](const auto& a, const auto& b){
        auto ka = key(a.first);
        auto kb = key(b.first);
        if(ka != kb) return ka < kb;
        return a.first < b.first;
    });

    for(const auto& kv : items){
        const std::string& id = kv.first;
        const auto& p = kv.second;
        const bool is_in = id.rfind("IN_", 0) == 0;

        // Convert logical px to framebuffer px
        float fb_x = (float)vp_x + (p.first - impl->v_min_x) * (float)vp_w / (impl->v_max_x - impl->v_min_x);
        float fb_y = (float)vp_y + (p.second - impl->v_min_y) * (float)vp_h / (impl->v_max_y - impl->v_min_y);

        // Center text
        SIZE text_size;
        auto wide_id = to_wide(id);
        GetTextExtentPoint32W(impl->hdc, wide_id.c_str(), (int)wide_id.size(), &text_size);
        int fb_px = (int)fb_x - text_size.cx / 2;
        int fb_py = (int)fb_y - text_size.cy / 2;

        gdi_draw_text_px(fb_px, fb_py, id, is_in ? in_color : out_color);
    }

    if(old_font){
        SelectObject(impl->hdc, old_font);
    }
}

void Renderer::draw_hud(const ScenarioEnv& env) const{
    int agents_alive = 0;
    for(const auto& c: env.cars) if(c.alive) agents_alive++;

    std::string line = "STEP: " + std::to_string(env.step_count) + " | AGENTS: " + std::to_string(agents_alive);
    if(env.traffic_flow){
        line += " | TRAFFIC: " + std::to_string((int)env.traffic_cars.size());
    }

    std::string lidar_line = "LIDAR: " + std::to_string((int)env.lidars.size());

    if(!env.lidars.empty()){
        const auto& lid = env.lidars[0];
        char buf2[128];
        std::snprintf(buf2, sizeof(buf2), " | RAYS: %d", (int)lid.distances.size());
        lidar_line += buf2;
    }

    if(!env.cars.empty() && env.cars[0].alive){
        float speed_ms = (env.cars[0].state.v * FPS) / SCALE;
        char buf[64];
        std::snprintf(buf, sizeof(buf), " | SPEED: %.1f M/S", speed_ms);
        lidar_line += buf;
    }

    if(impl && impl->hdc){
        gdi_draw_text_px(10, 10, line, RenderColors::HudTextRGB);
        gdi_draw_text_px(10, 34, lidar_line, RenderColors::HudTextRGB);
    }
}

#else // NOT _WIN32

void Renderer::draw_lane_ids(const ScenarioEnv& env) const {
    if (!imgui || !impl || !impl->window) return;

    // Square viewport logic to match render()
    int full_w = WIDTH;
    int full_h = HEIGHT;
    glfwGetFramebufferSize(impl->window, &full_w, &full_h);

    const int view = (full_w < full_h) ? full_w : full_h;
    const int vp_x = (full_w - view) / 2;
    const int vp_y = (full_h - view) / 2;

    float road_side = impl->v_max_x - impl->v_min_x; // This is now guaranteed square or close to it
    if (road_side < 1.0f) road_side = (float)WIDTH;

    for (const auto& kv : env.lane_layout.points) {
        const std::string& id = kv.first;
        const auto& p = kv.second;
        const bool is_in = id.rfind("IN_", 0) == 0;

        float fb_x = (float)vp_x + (p.first - impl->v_min_x) * (float)view / road_side;
        float fb_y = (float)vp_y + (p.second - impl->v_min_y) * (float)view / road_side;

        // Keep text fully inside the viewport
        const ImVec2 ts = ImGui::CalcTextSize(id.c_str());
        fb_x -= 0.5f * ts.x;
        fb_y -= 0.5f * ts.y;

        const float min_x_vp = float(vp_x);
        const float min_y_vp = float(vp_y);
        const float max_x_vp = float(vp_x + view) - ts.x;
        const float max_y_vp = float(vp_y + view) - ts.y;

        fb_x = std::max(min_x_vp, std::min(fb_x, max_x_vp));
        fb_y = std::max(min_y_vp, std::min(fb_y, max_y_vp));

        // Simple color coding
        const unsigned int rgba = is_in ? 0xFF3333FFu : 0xFF66AAFFu;
        imgui->add_text(fb_x, fb_y, id, rgba);
    }
}

void Renderer::draw_hud(const ScenarioEnv& env) const {
    if (!imgui || !impl || !impl->window) return;

    int full_w = WIDTH;
    int full_h = HEIGHT;
    glfwGetFramebufferSize(impl->window, &full_w, &full_h);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "STEP: %d | AGENTS: %d", env.step_count, (int)env.cars.size());
    imgui->add_text(10.0f, 10.0f, buf, 0xFFFFFFFFu);

    // Display status for the first agent to diagnose "stuck" issue
    if (!env.cars.empty()) {
        const auto& ego = env.cars[0];
        // We need to know the status from StepResult, but ScenarioEnv doesn't store it per car.
        // For debugging, we re-check off-road/line here or rely on 'alive' flag.
        std::string status = ego.alive ? "ALIVE" : "DEAD";
        
        std::snprintf(buf, sizeof(buf), "EGO 0: %s | POS: (%.1f, %.1f) | V: %.2f", 
                     status.c_str(), ego.state.x, ego.state.y, ego.state.v);
        imgui->add_text(10.0f, 35.0f, buf, ego.alive ? 0xFF00FF00u : 0xFF0000FFu);
    }
}

#endif // _WIN32


// ROUTE ---------------------------------------------------
void Renderer::draw_route(const ScenarioEnv& env) const{
    if(!impl) return;

    // --- Ego route (cyan) ---
    if(!env.cars.empty()) {
        const auto& car = env.cars[0];
        if(!car.path.empty()){
            glLineWidth(2.5f);
            glColor4f(RenderColors::RouteCyan.r, RenderColors::RouteCyan.g, RenderColors::RouteCyan.b, 0.8f);
            
            if (impl->view_mode == VIEW_2D) {
                glBegin(GL_LINE_STRIP);
                for(const auto& p : car.path){
                    glVertex2f(ndc_x(p.first, impl->v_min_x, impl->v_max_x),
                               ndc_y(p.second, impl->v_min_y, impl->v_max_y));
                }
                glEnd();
            } else {
                // 3D mode: draw on ground plane
                glDisable(GL_LIGHTING);
                glBegin(GL_LINE_STRIP);
                for(const auto& p : car.path){
                    glVertex3f(p.first, -0.4f, p.second);
                }
                glEnd();
                glEnable(GL_LIGHTING);
            }

            // Lookahead target point
            const int lookahead = 10;
            int target_idx = car.path_index + lookahead;
            if(target_idx < 0) target_idx = 0;
            if(target_idx >= (int)car.path.size()) target_idx = (int)car.path.size() - 1;

            const float tx = car.path[target_idx].first;
            const float ty = car.path[target_idx].second;

            if (impl->view_mode == VIEW_2D) {
                draw_circle_px(tx, ty, 4.0f, 10, RenderColors::TargetRed.r, RenderColors::TargetRed.g, RenderColors::TargetRed.b,
                               impl->v_min_x, impl->v_max_x, impl->v_min_y, impl->v_max_y);
            } else {
                // 3D Target Marker (Small quad on ground)
                glDisable(GL_LIGHTING);
                glColor3f(RenderColors::TargetRed.r, RenderColors::TargetRed.g, RenderColors::TargetRed.b);
                float s = 1.5f;
                glBegin(GL_QUADS);
                glVertex3f(tx - s, -0.35f, ty - s);
                glVertex3f(tx + s, -0.35f, ty - s);
                glVertex3f(tx + s, -0.35f, ty + s);
                glVertex3f(tx - s, -0.35f, ty + s);
                glEnd();
                glEnable(GL_LIGHTING);
            }
        }
    }

    // --- First NPC route (magenta) for debugging ---
    if(!env.traffic_cars.empty()){
        const auto& npc = env.traffic_cars[0];
        if(!npc.path.empty()){
            glLineWidth(2.0f);
            glColor4f(0.85f, 0.10f, 0.85f, 0.75f);
            glBegin(GL_LINE_STRIP);
            for(const auto& p : npc.path){
                glVertex2f(ndc_x(p.first, impl->v_min_x, impl->v_max_x),
                           ndc_y(p.second, impl->v_min_y, impl->v_max_y));
            }
            glEnd();

            // Mark NPC path_index and lookahead target
            int idx = npc.path_index;
            if(idx < 0) idx = 0;
            if(idx >= (int)npc.path.size()) idx = (int)npc.path.size() - 1;

            const float px = npc.path[idx].first;
            const float py = npc.path[idx].second;
            draw_circle_px(px, py, 4.0f, 10, 0.95f, 0.15f, 0.95f,
                           impl->v_min_x, impl->v_max_x, impl->v_min_y, impl->v_max_y);

            const int lookahead = 10;
            int tidx = idx + lookahead;
            if(tidx >= (int)npc.path.size()) tidx = (int)npc.path.size() - 1;
            const float tx = npc.path[tidx].first;
            const float ty = npc.path[tidx].second;
            draw_circle_px(tx, ty, 4.0f, 10, 1.0f, 0.30f, 1.0f,
                           impl->v_min_x, impl->v_max_x, impl->v_min_y, impl->v_max_y);
        }
    }
}

// CAR DRAW -----------------------------------------------
void Renderer::draw_cars(const ScenarioEnv& env) const{
    auto draw_one=[&](const Car& car, float r,float g,float b, bool npc){
        if(!car.alive) return;
        
        if (impl->view_mode == VIEW_2D) {
            float x=car.state.x; float y=car.state.y; float heading=car.state.heading;

            float len=CAR_LENGTH; float wid=CAR_WIDTH;
            float hl=len*0.5f; float hw=wid*0.5f;

            auto rot=[&](float lx,float ly){
                float vx = lx * std::cos(-heading) - ly * std::sin(-heading);
                float vy = lx * std::sin(-heading) + ly * std::cos(-heading);
                return std::pair<float,float>(x+vx, y+vy);
            };

            // Body
            glColor3f(r,g,b);
            std::array<std::pair<float,float>,4> body={{
                rot(+hl,+hw), rot(+hl,-hw), rot(-hl,-hw), rot(-hl,+hw)
            }};
            glBegin(GL_QUADS);
            for(const auto& p: body){ 
                glVertex2f(ndc_x(p.first, impl->v_min_x, impl->v_max_x), 
                           ndc_y(p.second, impl->v_min_y, impl->v_max_y)); 
            }
            glEnd();

            // Head marker
            float mr = npc ? RenderColors::TrafficHeadBlack.r : RenderColors::AgentHeadMarker.r;
            float mg = npc ? RenderColors::TrafficHeadBlack.g : RenderColors::AgentHeadMarker.g;
            float mb = npc ? RenderColors::TrafficHeadBlack.b : RenderColors::AgentHeadMarker.b;
            glColor3f(mr,mg,mb);

            float x0 = -hl + 0.70f*len;
            float x1 = -hl + 0.95f*len;
            float y0 = -hw + 2.0f;
            float y1 = +hw - 2.0f;

            std::array<std::pair<float,float>,4> head={{
                rot(x0,y0), rot(x1,y0), rot(x1,y1), rot(x0,y1)
            }};
            glBegin(GL_QUADS);
            for(const auto& p: head){ 
                glVertex2f(ndc_x(p.first, impl->v_min_x, impl->v_max_x), 
                           ndc_y(p.second, impl->v_min_y, impl->v_max_y)); 
            }
            glEnd();
        } else if (impl->meshes_loaded) {
            // OBJ Model Rendering
            glPushMatrix();
            glTranslatef(car.state.x, 0.0f, car.state.y);
            // Sim heading=0 is +X, Blender OBJ car head is often -Z or +Z.
            // Rotating -heading (in radians to degrees) + 90 degrees to align +X.
            glRotatef(car.state.heading * 57.29578f + 90.0f, 0, 1, 0);
            
            const auto& mesh = impl->mesh_car;
            glScalef(mesh.scale, mesh.scale, mesh.scale);
            
            mesh.draw(r, g, b);
            glPopMatrix();
        } else {
            // Fallback 3D Box if mesh failed to load
            float x=car.state.x; float y=car.state.y; float heading=car.state.heading;
            float len=CAR_LENGTH; float wid=CAR_WIDTH;
            float hl=len*0.5f; float hw=wid*0.5f;

            auto rot=[&](float lx,float ly){
                float vx = lx * std::cos(-heading) - ly * std::sin(-heading);
                float vy = lx * std::sin(-heading) + ly * std::cos(-heading);
                return std::pair<float,float>(x+vx, y+vy);
            };
            std::array<std::pair<float,float>,4> body={{
                rot(+hl,+hw), rot(+hl,-hw), rot(-hl,-hw), rot(-hl,+hw)
            }};

            glColor3f(r,g,b);
            float h = 4.0f;
            glBegin(GL_QUADS);
            for(const auto& p: body) glVertex3f(p.first, h, p.second);
            for(int i=0; i<4; ++i) {
                int next = (i+1)%4;
                glVertex3f(body[i].first, 0, body[i].second);
                glVertex3f(body[next].first, 0, body[next].second);
                glVertex3f(body[next].first, h, body[next].second);
                glVertex3f(body[i].first, h, body[i].second);
            }
            glEnd();
        }
    };

    static const std::array<std::array<float,3>,6> colors={{
        {231/255.f,76/255.f,60/255.f},{52/255.f,152/255.f,219/255.f},{46/255.f,204/255.f,113/255.f},
        {155/255.f,89/255.f,182/255.f},{241/255.f,196/255.f,15/255.f},{230/255.f,126/255.f,34/255.f}}};

    // Ego/agents
    for(size_t idx=0; idx<env.cars.size(); ++idx){
        auto col = colors[idx%colors.size()];
        draw_one(env.cars[idx], col[0], col[1], col[2], false);
    }

    // Traffic NPCs: gray body + black head marker
    for(const auto& npc : env.traffic_cars){
        draw_one(npc, RenderColors::TrafficBodyGray.r, RenderColors::TrafficBodyGray.g, RenderColors::TrafficBodyGray.b, true);
    }
}

// LIDAR ---------------------------------------------------
void Renderer::draw_lidar(const ScenarioEnv& env) const{
    const float line_r = RenderColors::LidarRayGreen.r;
    const float line_g = RenderColors::LidarRayGreen.g;
    const float line_b = RenderColors::LidarRayGreen.b;
    const float line_a = RenderColors::LidarRayGreen.a;

    const float hit_r = RenderColors::LidarHitRed.r;
    const float hit_g = RenderColors::LidarHitRed.g;
    const float hit_b = RenderColors::LidarHitRed.b;

    // Match Scenario/sensor.py: draw only hit rays
    const bool draw_all = false;

    for(size_t i=0;i<env.cars.size() && i<env.lidars.size();++i){
        if(!env.cars[i].alive) continue;
        const auto &lid=env.lidars[i];
        const auto &car=env.cars[i];
        float cx=car.state.x; float cy=car.state.y; float heading=car.state.heading;
        for(size_t k=0;k<lid.distances.size();++k){
            float dist=lid.distances[k];
            const bool hit = dist < lid.max_dist - 0.1f;
            if(!draw_all && !hit) continue;

            float ang=heading + lid.rel_angles[k];
            float ex=cx + dist*std::cos(ang);
            float ey=cy - dist*std::sin(ang);

            draw_line_px(cx, cy, ex, ey, 2.0f, line_r, line_g, line_b, line_a,
                         impl->v_min_x, impl->v_max_x, impl->v_min_y, impl->v_max_y);
            if(hit){
                draw_circle_px(ex, ey, 2.0f, 6, hit_r, hit_g, hit_b,
                               impl->v_min_x, impl->v_max_x, impl->v_min_y, impl->v_max_y);
            }
        }
    }
}
