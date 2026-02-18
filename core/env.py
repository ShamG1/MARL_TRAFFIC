"""Fast C++-backed scenario environment with Python-compatible API.

Rendering is delegated to the C++ OpenGL/GLFW renderer.

Notes:
- Multi-agent mode: returns obs (N,145), rewards (N,), terminated, truncated, info.
- traffic_flow mode (single-agent): returns obs (145,), reward scalar, terminated, truncated, info.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Union

import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from . import cpp_backend
except ImportError:
    import cpp_backend  # type: ignore

from .utils import build_lane_layout, ROUTE_MAP_BY_SCENARIO

# Local default reward config (mirrors Scenario/config.py)
DEFAULT_REWARD_CONFIG = {
    "use_team_reward": False,
    "traffic_flow": False,
    "reward_config": {
        "progress_scale": 10.0,
        "stuck_speed_threshold": 1.0,
        "stuck_penalty": -0.01,
        "crash_vehicle_penalty": -10.0,
        "crash_object_penalty": -5.0,
        "success_reward": 10.0,
        "action_smoothness_scale": -0.02,
        "team_alpha": 0.2,
    },
}


def _apply_reward_config(env: Any, reward_cfg: Dict[str, Any]) -> None:
    if not hasattr(env, "reward_config"):
        return
    rc = env.reward_config

    if "progress_scale" in reward_cfg:
        rc.k_prog = float(reward_cfg["progress_scale"])
    if "stuck_speed_threshold" in reward_cfg:
        rc.v_min_ms = float(reward_cfg["stuck_speed_threshold"])
    if "stuck_penalty" in reward_cfg:
        rc.k_stuck = float(reward_cfg["stuck_penalty"])
    if "crash_vehicle_penalty" in reward_cfg:
        rc.k_cv = float(reward_cfg["crash_vehicle_penalty"])
    if "crash_wall_penalty" in reward_cfg:
        rc.k_cw = float(reward_cfg["crash_wall_penalty"])
    if "crash_line_penalty" in reward_cfg:
        rc.k_cl = float(reward_cfg["crash_line_penalty"])
    # Fallback for old config
    if "crash_object_penalty" in reward_cfg:
        val = float(reward_cfg["crash_object_penalty"])
        if "crash_wall_penalty" not in reward_cfg: rc.k_cw = val
        if "crash_line_penalty" not in reward_cfg: rc.k_cl = val
    if "success_reward" in reward_cfg:
        rc.k_succ = float(reward_cfg["success_reward"])
    if "action_smoothness_scale" in reward_cfg:
        rc.k_sm = float(reward_cfg["action_smoothness_scale"])
    if "team_alpha" in reward_cfg:
        rc.alpha = float(reward_cfg["team_alpha"])


def _parse_num_lanes_from_scenario_name(name: str) -> int:
    # Special case for merge scenario: highway_merge_3lane uses 3 lanes
    if "highway_merge_3lane" == name:
        return 3
    # Accept patterns like: cross_2lane, t_junction_3lane_v2, roundabout_4lane
    m = re.findall(r"(?:^|_)(\d+)lane(?:$|_)", str(name))
    if m:
        return int(m[-1])

    # Allow scenarios without lane suffix by explicit mapping.
    # Keep this minimal: only handle known special cases here.
    if str(name) == "bottleneck":
        return 3

    raise ValueError(
        f"Cannot parse num_lanes from scenario_name={name!r}. Expected to contain like '_2lane', or be a known special-case scenario (e.g. 'bottleneck')."
    )


class ScenarioEnv:
    def __init__(self, config: Dict[str, Any] | None = None):
        if config is None:
            config = {}

        self.scenario_name = config.get("scenario_name", None)
        if not self.scenario_name:
            raise ValueError("config['scenario_name'] is required")

        self.num_lanes = _parse_num_lanes_from_scenario_name(str(self.scenario_name))

        self.traffic_flow = bool(config.get("traffic_flow", False))

        if self.traffic_flow:
            self.num_agents = 1
        else:
            self.num_agents = int(config.get("num_agents", 1))

        self.render_mode = config.get("render_mode", None)
        self.show_lane_ids = bool(config.get("show_lane_ids", False))
        self.show_lidar = bool(config.get("show_lidar", False))

        # Determine use_team_reward: default to True if num_agents > 1, unless traffic_flow is enabled.
        # Allow explicit override from config if present.
        if "use_team_reward" in config:
            use_team = bool(config["use_team_reward"])
        else:
            use_team = bool(self.num_agents > 1)

        if self.traffic_flow:
            use_team = False

        respawn = bool(config.get("respawn_enabled", True))
        max_steps = int(config.get("max_steps", 2000))

        self.ego_routes = config.get("ego_routes", None)
        if self.ego_routes is None:
            mapping = ROUTE_MAP_BY_SCENARIO.get(str(self.scenario_name), None)
            if mapping is None:
                raise RuntimeError(f"No route mapping defined for scenario_name={self.scenario_name!r}")

            routes = []
            for mp in mapping.values():
                for in_id, out_id in mp.items():
                    # Support both numeric indices and full string IDs
                    final_in = in_id if isinstance(in_id, str) else f"IN_{in_id}"
                    final_out = out_id if isinstance(out_id, str) else f"OUT_{out_id}"
                    routes.append((final_in, final_out))

            if not routes:
                raise RuntimeError(f"No valid routes generated for scenario_name={self.scenario_name!r}")

            self.ego_routes = [routes[i % len(routes)] for i in range(self.num_agents)]

        self.lane_layout = build_lane_layout(self.num_lanes)
        self.points = self.lane_layout["points"]

        if not cpp_backend.has_cpp_backend():
            raise RuntimeError(
                "SIM_MARL_ENV backend not available. Build it first, or run with the pure-Python environment. "
                "(cpp/build or cpp/build/Release must contain the SIM_MARL_ENV extension)"
            )

        self.env = cpp_backend.ScenarioEnv(self.num_lanes)
        self.env.set_scenario_name(str(self.scenario_name))

        # Inject explicit route intents so C++ does not infer intent from lane ids.
        mapping = ROUTE_MAP_BY_SCENARIO.get(str(self.scenario_name), None)
        if mapping is None:
            raise RuntimeError(f"No route mapping defined for scenario_name={self.scenario_name!r}")

        intent_ids = {"straight": 0, "left": 1, "right": 2}
        intent_items = []
        for turn_type, mp in mapping.items():
            intent_id = intent_ids.get(turn_type, 0)
            for in_idx, out_idx in mp.items():
                intent_items.append(((f"IN_{in_idx}", f"OUT_{out_idx}"), int(intent_id)))

        self.env.set_route_intents(intent_items)
        self.env.configure(use_team, respawn, max_steps)

        # Scenario bitmaps (required)
        scenarios_root = config.get(
            "scenarios_root",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenarios"),
        )
        scenario_dir = os.path.join(scenarios_root, str(self.scenario_name))
        drivable_png = os.path.join(scenario_dir, "drivable.png")
        yellowline_png = os.path.join(scenario_dir, "yellowline.png")
        lane_dashes_png = os.path.join(scenario_dir, "lane_dashes.png")
        lane_id_png = os.path.join(scenario_dir, "lane_id.png")
        ok = self.env.load_scenario_bitmaps(drivable_png, yellowline_png, lane_dashes_png, lane_id_png)
        if not ok:
            raise RuntimeError(
                f"Failed to load scenario bitmaps from {scenario_dir}. "
                "Expected drivable.png, yellowline.png, lane_id.png with resolution matching constants.h"
            )

        # traffic flow (single ego + NPC)
        self.traffic_density = float(config.get("traffic_density", 0.5))
        self.traffic_mode = config.get("traffic_mode", "stochastic")
        self.traffic_kmax = int(config.get("traffic_kmax", 20))

        self.env.configure_traffic(self.traffic_flow, self.traffic_density)
        self.env.set_traffic_mode(self.traffic_mode, self.traffic_kmax)

        traffic_routes = []
        for mp in mapping.values():
            for in_idx, out_idx in mp.items():
                final_in = in_idx if isinstance(in_idx, str) else f"IN_{in_idx}"
                final_out = out_idx if isinstance(out_idx, str) else f"OUT_{out_idx}"
                traffic_routes.append((final_in, final_out))
        self.env.configure_routes(traffic_routes)

        reward_cfg = config.get("reward_config", None)
        if reward_cfg is None:
            reward_cfg = DEFAULT_REWARD_CONFIG.get("reward_config", {})
        if isinstance(reward_cfg, dict):
            _apply_reward_config(self.env, reward_cfg)

        self.cars: List[cpp_backend.Car] = []
        self.traffic_cars: List[cpp_backend.Car] = []

        self.reset()


    def reset(self):
        self.env.reset()

        # Ensure ego_routes length matches num_agents
        if not self.ego_routes:
            raise RuntimeError("ego_routes is empty; cannot reset environment")
        if len(self.ego_routes) < self.num_agents:
            # Cycle routes if fewer routes than agents
            self.ego_routes = [self.ego_routes[i % len(self.ego_routes)] for i in range(self.num_agents)]

        for i in range(self.num_agents):
            start_id, end_id = self.ego_routes[i]
            self.env.add_car_with_route(start_id, end_id)
        self.cars = self.env.cars
        if self.traffic_flow:
            try:
                self.traffic_cars = list(self.env.traffic_cars)
            except Exception:
                self.traffic_cars = []
        obs = self._collect_obs()
        if self.traffic_flow:
            return obs[0], {}
        return obs, {}

    def _collect_obs(self) -> np.ndarray:
        # Zero-copy fast path (C++ returns a NumPy view)
        if hasattr(self.env, "get_observations_numpy"):
            try:
                obs = self.env.get_observations_numpy()
                return np.asarray(obs, dtype=np.float32)
            except Exception:
                pass

        # Fallback: flat buffer
        if hasattr(self.env, "get_observations_flat"):
            obs_flat = self.env.get_observations_flat()
            return np.asarray(obs_flat, dtype=np.float32).reshape(-1, 145)

        # Legacy fallback
        obs = self.env.get_observations()
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions: Union[np.ndarray, List[List[float]], List[float]], dt: float = 1.0 / 60.0):
        actions = np.asarray(actions, dtype=np.float32)

        # Accept both (2,) and (1,2) for single-agent manual control
        if self.traffic_flow:
            actions = actions.reshape(1, 2)
        else:
            if actions.ndim == 1:
                if actions.size == 2 and self.num_agents == 1:
                    actions = actions.reshape(1, 2)
                else:
                    raise ValueError(f"Expected actions shape (N,2) for multi-agent, got {actions.shape}")

        # Use optimized numpy step if available
        if hasattr(self.env, "step_numpy"):
            res = self.env.step_numpy(actions, float(dt))
        else:
            res = self.env.step(actions[:, 0].tolist(), actions[:, 1].tolist(), float(dt))

        if self.traffic_flow:
            try:
                self.traffic_cars = list(self.env.traffic_cars)
            except Exception:
                self.traffic_cars = []

        # Optimization: res.obs from C++ is often a copy. 
        # We prefer using the zero-copy observations buffer which is already updated in step().
        obs = self._collect_obs()

        rewards = np.asarray(res.rewards, dtype=np.float32)
        terminated = bool(res.terminated)
        truncated = bool(res.truncated)

        collisions = {int(res.agent_ids[i]): str(res.status[i]) for i in range(len(res.status))}

        info = {
            "step": int(res.step),
            "rewards": rewards.tolist() if not self.traffic_flow else float(rewards[0]) if len(rewards) else 0.0,
            "collisions": collisions,
            "agents_alive": int(getattr(res, "agents_alive", 0)),
            "terminated": terminated,
            "truncated": truncated,
            "done": list(res.done),
            "status": list(res.status),
        }

        if self.traffic_flow:
            return obs[0], float(rewards[0]) if len(rewards) else 0.0, terminated, truncated, info
        return obs, rewards, terminated, truncated, info

    def freeze_traffic(self, freeze: bool):
        """Freeze/unfreeze NPC refills (useful for MCTS rollouts)."""
        self.env.freeze_traffic(freeze)

    def render(
        self,
        show_lane_ids: bool | None = None,
        show_lidar: bool | None = None,
        show_connections: bool | None = None,
    ):
        if self.render_mode != "human":
            return
        if show_lane_ids is None:
            show_lane_ids = self.show_lane_ids
        if show_lidar is None:
            show_lidar = self.show_lidar
        if show_connections is None:
            show_connections = bool(getattr(self, "show_connections", False))
        self.env.render(bool(show_lane_ids), bool(show_lidar), bool(show_connections))

    def close(self):
        # C++ side owns the GLFW window; nothing to close here.
        pass


if __name__ == "__main__":
    env = ScenarioEnv({"num_agents": 6, "num_lanes": 3, "render_mode": "human", "respawn_enabled": True})
    env.reset()
    for _ in range(200):
        act = np.zeros((env.num_agents, 2), dtype=np.float32)
        env.step(act)
        env.render()
