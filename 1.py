# sanity_constant_traffic.py
# 用法（在项目根目录）:
#   python3 sanity_constant_traffic.py
#
# 目的：
# - constant 模式下：traffic_cars 的 size 恒定为 K = round(density * kmax)
# - freeze=True 时：不会补齐 dead slot（size 不变，alive_count 可能下降）
# - freeze=False 时：会补齐（size 不变，alive_count 会被拉回接近 K）

import sys
sys.path.insert(0, ".")

import numpy as np
import core.env as envmod


def alive_count(cars):
    return sum(1 for c in cars if getattr(c, "alive", False))


def main():
    cfg = {
        "scenario_name": "cross_2lane",
        "traffic_flow": True,
        "traffic_density": 0.5,   # density -> K
        "traffic_mode": "constant",
        "traffic_kmax": 20,       # K = round(0.5 * 20) = 10
        "render_mode": None,
        "max_steps": 200,
    }

    env = envmod.ScenarioEnv(cfg)
    obs, info = env.reset()

    # constant 模式下：reset 后 size 应恒定
    cpp = env.env
    K = int(round(float(cfg["traffic_density"]) * int(cfg["traffic_kmax"])))
    print("[reset] expected K =", K)
    print("[reset] traffic size =", len(cpp.traffic_cars), "alive =", alive_count(cpp.traffic_cars))

    # 连续 step 一段时间观察 size 是否恒定
    for t in range(1, 31):
        act = np.zeros((env.num_agents, 2), dtype=np.float32)
        obs, r, term, trunc, info = env.step(act)
        if t in (1, 2, 5, 10, 20, 30):
            print(f"[run t={t:02d}] size={len(cpp.traffic_cars)} alive={alive_count(cpp.traffic_cars)}")

    # 开启 freeze：不补齐 dead slot
    cpp.freeze_traffic(True)
    print("\n[freeze=True] enabled")
    for t in range(1, 31):
        act = np.zeros((env.num_agents, 2), dtype=np.float32)
        obs, r, term, trunc, info = env.step(act)
        if t in (1, 2, 5, 10, 20, 30):
            print(f"[freeze t={t:02d}] size={len(cpp.traffic_cars)} alive={alive_count(cpp.traffic_cars)}")

    # 关闭 freeze：恢复补齐
    cpp.freeze_traffic(False)
    print("\n[freeze=False] enabled (refill resumes)")
    for t in range(1, 31):
        act = np.zeros((env.num_agents, 2), dtype=np.float32)
        obs, r, term, trunc, info = env.step(act)
        if t in (1, 2, 5, 10, 20, 30):
            print(f"[refill t={t:02d}] size={len(cpp.traffic_cars)} alive={alive_count(cpp.traffic_cars)}")


if __name__ == "__main__":
    main()