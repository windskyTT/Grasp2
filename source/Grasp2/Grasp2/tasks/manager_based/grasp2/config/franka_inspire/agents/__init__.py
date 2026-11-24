# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 标识 Franka + Inspire 灵巧手任务的算法配置包

"""
| 配置文件 | 主要用途 |
|--------|--------|
| `rsl_rl_ppo_cfg.py` | RSL-RL 的 PPO 配置（实验名 franka_inspire_grasp2） |

目前只提供 RSL-RL 配置；如需使用 rl_games / skrl / sb3 训练，需要另外添加对应的配置文件，
并在 ../__init__.py 的 gym.register 中注册相应的 *_cfg_entry_point。
"""