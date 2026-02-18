#pragma once
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include <map>

#include "Car.h"
#include "Lidar.h"
#include "RoadGeometry.h"
#include "RoadMask.h"
#include "LineMask.h"
#include "RouteGen.h"
#include "constants.h"
#include "Reward.h"
#include "BitmapMask.h"
#include "EnvState.h"

#ifdef DRIVESIMX_ENABLE_RENDER
#include "Renderer.h"
#endif

// Match Scenario/config.py constants
constexpr int NEIGHBOR_COUNT = 5;

#ifdef DRIVESIMX_ENABLE_RENDER
class Renderer;
#endif

class ScenarioEnv {
public:
    ~ScenarioEnv();
    // Config
    int num_lanes;
    bool use_team_reward{false};
    bool respawn_enabled{true};
    int max_steps{2000};
    RewardConfig reward_config;

    // Scenario (Bitmap support)
    std::string scenario_name;
    bool use_bitmap_scenario{false};
    BitmapMask bitmap_road;
    BitmapMask bitmap_line;
    BitmapMask bitmap_dash;
    BitmapMask bitmap_lane;

    // Route Intent Map: (start_id, end_id) -> intent_id
    std::map<std::pair<std::string, std::string>, int> route_intents;

    // Traffic flow (NPC) - mode1: single ego + NPCs
    bool traffic_flow{false};
    float traffic_density{0.5f};

    enum class TrafficMode : int {
        STOCHASTIC = 0, // arrival-rate spawn + erase (default)
        CONSTANT = 1    // fixed-size NPC slots; no erase; optional refill
    };

    // Traffic behavior mode
    TrafficMode traffic_mode{TrafficMode::STOCHASTIC};

    // Maps traffic_density to fixed NPC slot count: K = round(traffic_density * traffic_kmax)
    int traffic_kmax{20};

    // When true in CONSTANT mode, do not refill dead NPC slots (useful for MCTS rollouts)
    bool traffic_freeze{false};

    // State
    LaneLayout lane_layout;
    RoadGeometry geom;
    RoadMask road_mask;
    LineMask line_mask;

    // Ego agents
    std::vector<Car> cars;
    std::vector<Lidar> lidars;
    std::vector<long long> agent_ids;

    // NPC traffic
    std::vector<Car> traffic_cars;
    std::vector<Lidar> traffic_lidars;
    std::vector<std::pair<std::string, std::string>> traffic_routes;

    long long next_agent_id{1};
    int step_count{0};

    explicit ScenarioEnv(int num_lanes_ = 3)
        : num_lanes(num_lanes_),
          lane_layout(build_lane_layout_cpp(num_lanes_)),
          geom(num_lanes_),
          road_mask(num_lanes_),
          line_mask(num_lanes_) {
        init_traffic_routes();
    }

    void configure(bool use_team, bool respawn, int max_s);

    // Enable/disable traffic flow and set density (arrival rate)
    void configure_traffic(bool enabled, float density);

    // High-level traffic API: two modes for usability
    // mode: "stochastic" (default) or "constant"
    // kmax: maps density to fixed NPC slots K = round(density * kmax) when mode=="constant"
    void set_traffic_mode(const std::string& mode, int kmax = 20);
    void freeze_traffic(bool freeze);

    // Configure routes for NPCs from Python
    void configure_routes(const std::vector<std::pair<std::string, std::string>>& routes);

    void reset();

    // Scenario assets
    // drivable_png: 255=road, 0=obstacle
    // yellowline_png: 255=line, 0=background
    // lane_id_png: 0=none, 1..255=lane id
    void set_scenario_name(const std::string& name) { scenario_name = name; }

    // Set explicit route intents from Python.
    // Each item: ((start_id, end_id), intent_id) where intent_id: 0=straight,1=left,2=right
    void set_route_intents(const std::vector<std::pair<std::pair<std::string,std::string>, int>>& items);

    bool load_scenario_bitmaps(const std::string& drivable_png,
                              const std::string& yellowline_png,
                              const std::string& dash_png,
                              const std::string& lane_id_png);

    void add_car_with_route(const std::string& start_id, const std::string& end_id);

    StepResult step(const std::vector<float>& throttles,
                    const std::vector<float>& steerings,
                    float dt = DT_DEFAULT);

    std::vector<std::vector<float>> get_observations() const;

    // Optimized API: Returns a flat contiguous buffer (num_agents * obs_dim).
    // Exposed to Python as a NumPy array for fewer allocations/copies.
    std::vector<float> get_observations_flat() const;

    // Direct access to internal observation buffer for zero-copy
    const float* get_obs_buffer_data() const { return obs_buffer.data(); }
    size_t get_obs_buffer_size() const { return obs_buffer.size(); }
    void update_observations_buffer();

    // CTDE: fixed-size centralized state for a given agent (ego + nearest-K egos).
    // Encoding uses ONLY ego agents (no NPCs).
    // Format: [ego_state(6), neigh1_state(6), neigh2_state(6), neigh3_state(6)]
    // where each state = [x_norm, y_norm, v_norm, heading_norm, intention, alive]
    std::vector<float> get_global_state(int agent_index, int k_nearest = 3) const;

    // Snapshot API for fast MCTS rollbacks
    EnvState get_state() const;
    void set_state(const EnvState& s);

#ifdef DRIVESIMX_ENABLE_RENDER
    void render(bool show_lane_ids = false, bool show_lidar = false, bool show_connections = false);
    void set_view_mode(int mode);
    int get_view_mode() const;

    // GLFW input/window helpers (available only after first render() creates a window)
    bool window_should_close() const;
    void poll_events() const;
    bool key_pressed(int glfw_key) const;
#else
    // Headless build stubs
    void render(bool show_lane_ids = false, bool show_lidar = false, bool show_connections = false);
    bool window_should_close() const;
    void poll_events() const;
    bool key_pressed(int glfw_key) const;
#endif

private:
    // Rendering
#ifdef DRIVESIMX_ENABLE_RENDER
    bool render_enabled{false};
    std::unique_ptr<Renderer> renderer; // allocated only when render_enabled
#endif

    void init_traffic_routes();
    void update_traffic_flow(float dt);
    void try_spawn_traffic_car();
    bool is_spawn_blocked(float sx, float sy) const;
    bool is_arrived(const Car& car, float tol = 20.0f) const;
    bool is_out_of_screen(const Car& car, float margin = 100.0f) const;

    // Persistent buffer for zero-copy observations
    mutable std::vector<float> obs_buffer;
};
