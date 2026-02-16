# This file is a local copy of necessary components from the 'Scenario' package
# to make C_MCTS a self-contained module.
# === From Scenario/config.py ===
WIDTH, HEIGHT = 1000, 1000
SCALE = 12
LANE_WIDTH_M = 3.5
LANE_WIDTH_PX = int(LANE_WIDTH_M * SCALE)



OBS_DIM = 135

# Default reward config (can be overridden via config dict)
DEFAULT_REWARD_CONFIG = {
    'use_team_reward': False,  # Use team reward mixing (for multi-agent)
    'traffic_flow': False,      # If True, forces individual reward (single-agent with traffic)
    'reward_config': {
        'progress_scale': 10.0, 
        'stuck_speed_threshold': 1.0,  # m/s
        'stuck_penalty': -0.01,
        'crash_vehicle_penalty': -10.0,
        'crash_wall_penalty': -5.0,   # Off-road / wall
        'crash_line_penalty': -0.1,   # Yellow line crossing (lighter than wall)
        'crash_object_penalty': -5.0,  # Legacy fallback: applies to both if specific keys missing
        'success_reward': 10.0,
        'action_smoothness_scale': -0.02,
        'team_alpha': 0.2,
    }
}
# === From Scenario/env.py ===
ROUTE_MAP_BY_SCENARIO = {
    "cross_2lane": {
        "straight": {2: 6, 4: 8, 6: 2, 8: 4},
        "left": {1: 3, 3: 5, 5: 7, 7: 1},
    },
    "cross_3lane": {
        "straight": {2: 8, 5: 11, 8: 2, 11: 5},
        "right": {3: 12, 6: 3, 9: 6, 12: 9},
        "left": {1: 4, 4: 7, 7: 10, 10: 1},
    },
    "T_2lane": {
        "straight": {2: 6, 5: 1},
        "right": {4: 2, 6: 4},
        "left": {1: 3, 3: 5},
    },
    "T_3lane": {
        "straight": {2: 8, 3: 9, 7: 1, 8: 2},
        "left": {1: 4, 4: 7, 5: 8},
        "right": {6: 3, 9: 6, 5: 2},
    },
    "highway_2lane": {
        "straight": {1: 1, 2: 2},
    },
    "highway_4lane": {
        "straight": {1: 1, 2: 2, 3: 3, 4: 4},
    },
    "roundabout_2lane": {
        "straight": {2: 6, 4: 8, 6: 2, 8: 4},
        "left": {1: 3, 3: 5, 5: 7, 7: 1},
    },
    "roundabout_3lane": {
        "straight": {2: 8, 5: 11, 8: 2, 11: 5},
        "right": {3: 12, 6: 3, 9: 6, 12: 9},
        "left": {1: 4, 4: 7, 7: 10, 10: 1},
    },
    "onrampmerge_3lane": {
        "straight": {1: 1, 2: 2},
        "ramp": {"IN_RAMP_1": "OUT_2"},
    },
    "bottleneck": {
        "straight": {1: 1, 2: 2, 3: 3},
    },
}



# === From Scenario/agent.py ===
def build_lane_layout(num_lanes: int):
    dir_order = ['N', 'E', 'S', 'W']
    points = {}
    in_by_dir = {d: [] for d in dir_order}
    out_by_dir = {d: [] for d in dir_order}
    dir_of = {}
    idx_of = {}
    MARGIN = 30
    CX, CY = WIDTH // 2, HEIGHT // 2

    for d_idx, d in enumerate(dir_order):
        for j in range(num_lanes):
            offset = LANE_WIDTH_PX * (0.5 + j)
            in_name = f"IN_{d_idx * num_lanes + j + 1}"
            out_name = f"OUT_{d_idx * num_lanes + j + 1}"

            if d == 'N':
                points[in_name] = (CX - offset, MARGIN)
                points[out_name] = (CX + offset, MARGIN)
            elif d == 'S':
                points[in_name] = (CX + offset, HEIGHT - MARGIN)
                points[out_name] = (CX - offset, HEIGHT - MARGIN)
            elif d == 'E':
                points[in_name] = (WIDTH - MARGIN, CY - offset)
                points[out_name] = (WIDTH - MARGIN, CY + offset)
            else:  # 'W'
                points[in_name] = (MARGIN, CY + offset)
                points[out_name] = (MARGIN, CY - offset)

            in_by_dir[d].append(in_name)
            out_by_dir[d].append(out_name)
            dir_of[in_name] = d
            dir_of[out_name] = d
            idx_of[in_name] = j
            idx_of[out_name] = j

    return {
        'points': points,
        'in_by_dir': in_by_dir,
        'out_by_dir': out_by_dir,
        'dir_of': dir_of,
        'idx_of': idx_of,
        'dir_order': dir_order,
    }
