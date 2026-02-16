from re import T
import numpy as np
import random
import sys
import os
import time

# Make sure repo root is available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import ScenarioEnv, ROUTE_MAP_BY_SCENARIO

# mapping: cross_2lane, cross_3lane, roundabout_2lane, roundabout_3lane,
#          T_2lane, T_3lane, highway_2lane, 
#          highway_4lane, onrampmerge_3lane, bottleneck

def main():
    config = {
        'traffic_flow': False,
        'traffic_density': 1,
        'num_agents': 1,
        'scenario_name': 'bottleneck',
        'render_mode': 'human',
        'max_steps': 2000,
        'respawn_enabled': True,
        'show_lane_ids': False,
        'show_lidar': True,
    }

    env = ScenarioEnv(config)

    # --- Route selection logic ---
    mapping = ROUTE_MAP_BY_SCENARIO.get(str(config.get('scenario_name')), None)
    if mapping is None:
        raise RuntimeError(f"No route mapping defined for scenario_name={config.get('scenario_name')!r}")

    all_routes = []
    for mp in mapping.values():
        for in_id, out_id in mp.items():
            start = in_id if isinstance(in_id, str) else f"IN_{in_id}"
            end = out_id if isinstance(out_id, str) else f"OUT_{out_id}"
            all_routes.append((start, end))

    def choose_random_route():
        return random.choice(all_routes)

    env.ego_routes = [choose_random_route()]
    obs, info = env.reset()

    print("=" * 60)
    print("Manual Control Test (Scenario C++ backend)")
    print("=" * 60)
    print("Controls:")
    print("  UP/DOWN arrows: Throttle")
    print("  LEFT/RIGHT arrows: Steering")
    print("  R: Reset environment with a new random route")
    print("  L: Toggle Lidar visualization")
    print("  ESC/Q: Quit")
    print("=" * 60)

    total_reward = 0.0
    running = True
    show_lidar = True
    print_obs = False

    # GLFW key codes (same as GLFW_KEY_*)
    KEY_UP = 265
    KEY_DOWN = 264
    KEY_LEFT = 263
    KEY_RIGHT = 262
    KEY_R = 82
    KEY_L = 76
    KEY_O = 79
    KEY_Q = 81
    KEY_ESC = 256

    # One-time: make sure render window is created before polling keys
    env.render(show_lane_ids=config.get('show_lane_ids', False), show_lidar=show_lidar)

    last_toggle_l = 0.0
    last_toggle_o = 0.0

    target_dt = 1.0 / 60.0
    last_t = time.perf_counter()

    intent_labels = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}

    def ego_intention_label():
        try:
            intent = env.env.cars[0].intention
        except Exception:
            return "UNKNOWN"
        return intent_labels.get(int(intent), f"UNKNOWN({intent})")

    def print_obs_snapshot(obs_array):
        if obs_array is None:
            print("Obs not yet available.")
            return
        flat = np.asarray(obs_array).flatten()
        print(f"Obs shape: {flat.shape}, min: {flat.min():.4f}, max: {flat.max():.4f}, mean: {flat.mean():.4f}")
        preview_len = min(20, flat.size)
        preview = np.array2string(flat[:preview_len], precision=3, separator=", ")
        if flat.size > preview_len:
            print(f"Obs preview (first {preview_len}): {preview} ...")
        else:
            print(f"Obs values: {preview}")

    while running:
        now_t = time.perf_counter()
        frame_dt = now_t - last_t
        last_t = now_t

        # Avoid giant dt after breakpoints / window dragging
        if frame_dt > 0.25:
            frame_dt = 0.25

        # Pump OS events so key states and close button work
        env.env.poll_events()

        if env.env.window_should_close():
            break

        # Edge-trigger toggles
        now = time.time()
        if env.env.key_pressed(KEY_R):
            env.ego_routes = [choose_random_route()]
            obs, info = env.reset()
            total_reward = 0.0
            print(f"Environment reset! New route: {env.ego_routes[0]}, intention: {ego_intention_label()}")
            time.sleep(0.15)

        if env.env.key_pressed(KEY_L) and (now - last_toggle_l) > 0.2:
            show_lidar = not show_lidar
            last_toggle_l = now

        if env.env.key_pressed(KEY_O) and (now - last_toggle_o) > 0.2:
            print_obs = not print_obs
            print(f"Observation logging {'ENABLED' if print_obs else 'DISABLED'}.")
            if print_obs:
                print(f"Ego intention: {ego_intention_label()}")
                print_obs_snapshot(obs)
            last_toggle_o = now

        if env.env.key_pressed(KEY_ESC) or env.env.key_pressed(KEY_Q):
            running = False

        throttle = 0.3 if env.env.key_pressed(KEY_UP) else -0.5 if env.env.key_pressed(KEY_DOWN) else 0.0
        steer = 1.0 if env.env.key_pressed(KEY_LEFT) else -1.0 if env.env.key_pressed(KEY_RIGHT) else 0.0

        action = np.array([throttle, steer], dtype=np.float32)

        # Advance simulation time using real elapsed time, but integrate with fixed substeps
        remaining = frame_dt
        obs = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        while remaining > 1e-9 and not (terminated or truncated):
            dt = remaining if remaining < target_dt else target_dt
            remaining -= dt
            obs, r, terminated, truncated, info = env.step(action, dt=dt)
            reward += float(r)
        total_reward += float(reward)
        done = terminated or truncated

        if print_obs:
            print(f"Ego intention: {ego_intention_label()}")
            print_obs_snapshot(obs)

        if done:
            print(f"Episode ended: {info.get('collisions', {})}, Total Reward: {total_reward:.4f}")
            env.ego_routes = [choose_random_route()]
            obs, info = env.reset()
            total_reward = 0.0

        env.render(show_lane_ids=config.get('show_lane_ids', False), show_lidar=show_lidar)

        # No pygame overlay; HUD handled by C++ renderer

    env.close()
    print("Test completed!")


if __name__ == '__main__':
    main()
