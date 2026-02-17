#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Car.h"
#include "ScenarioEnv.h"
#include "Lidar.h"
#include "Reward.h"
#include "EnvState.h"
#include "mcts_search.h"

namespace py = pybind11;

PYBIND11_MODULE(DRIVESIMX_ENV, m) {
    m.doc() = "C++ backend for high-speed Scenario MCTS";

    py::class_<State>(m, "State")
        .def(py::init<>())
        .def_readwrite("x", &State::x)
        .def_readwrite("y", &State::y)
        .def_readwrite("v", &State::v)
        .def_readwrite("heading", &State::heading);

    py::class_<Car>(m, "Car")
        .def(py::init<>())
        .def_readwrite("state", &Car::state)
        .def_readwrite("length", &Car::length)
        .def_readwrite("width", &Car::width)
        .def_readwrite("alive", &Car::alive)
        .def_readwrite("intention", &Car::intention)
        .def_readwrite("path_index", &Car::path_index)
        .def_readwrite("path", &Car::path)
        .def("update", &Car::update)
        .def("check_collision", &Car::check_collision);

    py::class_<RewardConfig>(m, "RewardConfig")
        .def(py::init<>())
        .def_readwrite("k_prog", &RewardConfig::k_prog)
        .def_readwrite("v_min_ms", &RewardConfig::v_min_ms)
        .def_readwrite("k_stuck", &RewardConfig::k_stuck)
        .def_readwrite("k_cv", &RewardConfig::k_cv)
        .def_readwrite("k_cw", &RewardConfig::k_cw)
        .def_readwrite("k_cl", &RewardConfig::k_cl)
        .def_readwrite("k_succ", &RewardConfig::k_succ)
        .def_readwrite("k_sm", &RewardConfig::k_sm)
        .def_readwrite("alpha", &RewardConfig::alpha);

    py::class_<StepResult>(m, "StepResult")
        .def(py::init<>())
        .def_readwrite("obs", &StepResult::obs)
        .def_readwrite("rewards", &StepResult::rewards)
        .def_readwrite("done", &StepResult::done)
        .def_readwrite("status", &StepResult::status)
        .def_readwrite("agent_ids", &StepResult::agent_ids)
        .def_readwrite("agents_alive", &StepResult::agents_alive)
        .def_readwrite("terminated", &StepResult::terminated)
        .def_readwrite("truncated", &StepResult::truncated)
        .def_readwrite("step", &StepResult::step);

    py::class_<EnvState>(m, "EnvState")
        .def(py::init<>())
        .def_readwrite("cars", &EnvState::cars)
        .def_readwrite("traffic_cars", &EnvState::traffic_cars)
        .def_readwrite("agent_ids", &EnvState::agent_ids)
        .def_readwrite("next_agent_id", &EnvState::next_agent_id)
        .def_readwrite("step_count", &EnvState::step_count);

    py::class_<ScenarioEnv>(m, "ScenarioEnv")
        .def(py::init<int>(), py::arg("num_lanes") = 3)
        .def_readwrite("cars", &ScenarioEnv::cars)
        .def_readwrite("traffic_cars", &ScenarioEnv::traffic_cars)
        .def_readwrite("lidars", &ScenarioEnv::lidars)
        .def_readwrite("step_count", &ScenarioEnv::step_count)
        .def_readwrite("reward_config", &ScenarioEnv::reward_config)
        .def("set_scenario_name", &ScenarioEnv::set_scenario_name, py::arg("name"))
        .def("set_route_intents", &ScenarioEnv::set_route_intents, py::arg("items"))
        .def("configure", &ScenarioEnv::configure, py::arg("use_team"), py::arg("respawn"), py::arg("max_steps"))
        .def("configure_traffic", &ScenarioEnv::configure_traffic, py::arg("enabled"), py::arg("density"))
        .def("set_traffic_mode", &ScenarioEnv::set_traffic_mode, py::arg("mode"), py::arg("kmax") = 20)
        .def("freeze_traffic", &ScenarioEnv::freeze_traffic, py::arg("freeze"))
        .def("configure_routes", &ScenarioEnv::configure_routes, py::arg("routes"))
        .def("reset", &ScenarioEnv::reset)
        .def("add_car_with_route", &ScenarioEnv::add_car_with_route, py::arg("start_id"), py::arg("end_id"))
        .def("load_scenario_bitmaps", &ScenarioEnv::load_scenario_bitmaps, py::arg("drivable_png"), py::arg("yellowline_png"), py::arg("dash_png"), py::arg("lane_id_png"))
        .def("step", &ScenarioEnv::step, py::arg("throttles"), py::arg("steerings"), py::arg("dt") = 1.0/60.0)
        .def("get_observations", &ScenarioEnv::get_observations)
        .def("get_observations_flat", &ScenarioEnv::get_observations_flat)
        .def("get_observations_numpy", [](ScenarioEnv& env) {
            env.update_observations_buffer();
            const size_t n = env.cars.size();
            const int obs_dim = 145;
            auto base = py::cast(&env);
            return py::array_t<float>(
                {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(obs_dim)},
                {static_cast<py::ssize_t>(obs_dim * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
                env.get_obs_buffer_data(),
                base
            );
        })
        .def("get_global_state", &ScenarioEnv::get_global_state, py::arg("agent_index"), py::arg("k_nearest") = 3)
        .def("get_state", &ScenarioEnv::get_state)
        .def("set_state", &ScenarioEnv::set_state, py::arg("state"))
        .def("render", &ScenarioEnv::render, py::arg("show_lane_ids") = false, py::arg("show_lidar") = false)
        .def("set_view_mode", &ScenarioEnv::set_view_mode, py::arg("mode"))
        .def("get_view_mode", &ScenarioEnv::get_view_mode)
        .def("window_should_close", &ScenarioEnv::window_should_close)
        .def("poll_events", &ScenarioEnv::poll_events)
        .def("key_pressed", &ScenarioEnv::key_pressed, py::arg("glfw_key"));

    py::class_<Lidar>(m, "Lidar")
        .def(py::init<>())
        .def_readwrite("rays", &Lidar::rays)
        .def_readwrite("fov_deg", &Lidar::fov_deg)
        .def_readwrite("max_dist", &Lidar::max_dist)
        .def_readwrite("step_size", &Lidar::step_size)
        .def_readwrite("distances", &Lidar::distances)
        .def_readwrite("rel_angles", &Lidar::rel_angles)
        .def("normalized", &Lidar::normalized);
    
    m.def("mcts_search_seq", &mcts_search_seq,
        py::arg("env"),
        py::arg("root_state"),
        py::arg("root_obs_seq"),
        py::arg("seq_len"),
        py::arg("obs_dim"),
        py::arg("infer_fn"),
        py::arg("agent_index"),
        py::arg("num_simulations"),
        py::arg("num_action_samples"),
        py::arg("rollout_depth"),
        py::arg("c_puct"),
        py::arg("temperature"),
        py::arg("gamma"),
        py::arg("dirichlet_alpha"),
        py::arg("dirichlet_eps"),
        py::arg("seed")
    );

    m.def("mcts_search", &mcts_search,
        py::arg("env"),
        py::arg("root_state"),
        py::arg("root_obs"),
        py::arg("infer_fn"),
        py::arg("agent_index"),
        py::arg("num_simulations"),
        py::arg("num_action_samples"),
        py::arg("rollout_depth"),
        py::arg("c_puct"),
        py::arg("temperature"),
        py::arg("gamma"),
        py::arg("dirichlet_alpha"),
        py::arg("dirichlet_eps"),
        py::arg("seed")
    );

    m.def("mcts_search_lstm", &mcts_search_lstm,
        py::arg("env"),
        py::arg("root_state"),
        py::arg("root_obs"),
        py::arg("infer_policy_value"),
        py::arg("infer_next_hidden"),
        py::arg("root_h"),
        py::arg("root_c"),
        py::arg("lstm_hidden_dim"),
        py::arg("agent_index"),
        py::arg("num_simulations"),
        py::arg("num_action_samples"),
        py::arg("rollout_depth"),
        py::arg("c_puct"),
        py::arg("temperature"),
        py::arg("gamma"),
        py::arg("dirichlet_alpha"),
        py::arg("dirichlet_eps"),
        py::arg("seed")
    );

    m.def("mcts_search_lstm_torchscript", &mcts_search_lstm_torchscript,
        py::arg("env"),
        py::arg("root_state"),
        py::arg("root_obs_seq"),
        py::arg("seq_len"),
        py::arg("obs_dim"),
        py::arg("model_path"),
        py::arg("root_h"),
        py::arg("root_c"),
        py::arg("lstm_hidden_dim"),
        py::arg("agent_index"),
        py::arg("num_simulations"),
        py::arg("num_action_samples"),
        py::arg("rollout_depth"),
        py::arg("c_puct"),
        py::arg("temperature"),
        py::arg("gamma"),
        py::arg("dirichlet_alpha"),
        py::arg("dirichlet_eps"),
        py::arg("seed")
    );
}