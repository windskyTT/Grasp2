# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 注册Franka Lift5个任务

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control（关节位置控制） FRANKA_PANDA_CFG
##

gym.register(
    id="Isaac-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={ #创建额外的环境初始化及配置入口参数字典
        # 环境配置入口 配置定位到：Grasp2.tasks.manager_based.lift.config.franka.joint_pos_env_cfg:FrankaCubeLiftEnvCfg
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg", 
        # 算法配置入口 RSL-RL 配置定位到：Grasp2.tasks.manager_based.lift.config.franka.agents.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg", 
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True, # 关闭 Gymnasium 默认环境检查包装逻辑
)

# 关节位置控制，使用play环境配置  FRANKA_PANDA_CFG
gym.register(
    id="Isaac-Lift-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##
# 绝对位姿控制 FRANKA_PANDA_HIGH_PD_CFG
gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
# 注意，本段只注册环境配置，并没有填写 RSL-RL 等算法配置入口。
# 不能简单认为所有已注册的 IK 环境都能直接使用同一个默认 PPO 训练配置。

gym.register(
    id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##
# 相对位姿控制 FRANKA_PANDA_HIGH_PD_CFG
gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaCubeLiftEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc.json",
        #BC（Behavioral Cloning，行为克隆） 属于模仿学习（Imitation Learning）的典型方法
        # 它与 PPO 的训练方式不同：
        # - PPO 主要通过与环境交互获得奖励，进行策略优化。
        # - BC 主要利用已有示范数据，学习模仿示范动作
    },
    disable_env_checker=True,
)
