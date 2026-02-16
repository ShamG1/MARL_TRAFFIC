#include "ScenarioEnv.h"

#ifdef SIM_MARL_ENABLE_RENDER
#include "Renderer.h"

void ScenarioEnv::render(bool show_lane_ids, bool show_lidar){
    if(!render_enabled){
        render_enabled=true;
    }
    if(!renderer){
        renderer=std::make_unique<Renderer>();
    }
    if(renderer && renderer->ok()){
        renderer->render(*this, show_lane_ids, show_lidar);
    }
}

bool ScenarioEnv::window_should_close() const {
    if(!renderer) return true;
    return renderer->window_should_close();
}

void ScenarioEnv::poll_events() const {
    if(!renderer) return;
    renderer->poll_events();
}

bool ScenarioEnv::key_pressed(int glfw_key) const {
    if(!renderer) return false;
    return renderer->key_pressed(glfw_key);
}
#else
// Headless build stubs
void ScenarioEnv::render(bool, bool) {}
bool ScenarioEnv::window_should_close() const { return true; }
void ScenarioEnv::poll_events() const {}
bool ScenarioEnv::key_pressed(int) const { return false; }
#endif