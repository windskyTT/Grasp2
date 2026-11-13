# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents # 导入当前任务目录下的 agents 配置包

##
# Register Gym environments.
##


gym.register( # 调用 Gymnasium 的环境注册函数
    id="Template-Grasp2-v0", # 指定任务在 Gymnasium 注册表中的唯一识别名称
    entry_point="isaaclab.envs:ManagerBasedRLEnv", # 指定创建环境时使用的环境类
    disable_env_checker=True, # 关闭 Gymnasium 在环境创建时默认进行的环境检查包装逻辑
    kwargs={ # 开始定义环境相关的额外关键字参数字典
        "env_cfg_entry_point": f"{__name__}.grasp2_env_cfg:Grasp2EnvCfg", # 指定环境配置类的导入路径
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml", #指定 RL-Games 算法配置文件路径
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg", #指定 RSL-RL 算法运行器配置类路径
    },
)

"""
entry_point 和 env_cfg_entry_point 有什么区别
entry_point="isaaclab.envs:ManagerBasedRLEnv" 指定的是 环境实现类（Environment Implementation Class）。
"env_cfg_entry_point": f"{__name__}.grasp2_env_cfg:Grasp2EnvCfg"指定的是 环境配置类（Environment Configuration Class）

ManagerBasedRLEnv 是一套能够运行强化学习环境的通用框架。

Grasp2EnvCfg 则告诉这个通用框架：具体要运行什么场景、什么任务、使用什么机器人。
"""