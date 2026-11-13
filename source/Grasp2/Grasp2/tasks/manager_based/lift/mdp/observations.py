# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Observation（观测）：策略从环境中获得什么信息。
Coordinate Frame（坐标系）：如何描述物体相对于机器人的位置。
Coordinate Transformation（坐标变换）：如何从世界坐标转换到机器人根坐标系。
Quaternion（四元数）：如何表示和处理机器人姿态。
PyTorch Tensor（张量）：如何同时处理几千个并行环境的数据。
MDP（马尔可夫决策过程）：观测函数如何参与 PPO 强化学习训练。
"""
#主要作用是：计算被抓取物体相对于机器人根坐标系的位置，并将结果作为强化学习策略的一项观测

from __future__ import annotations #类型注解（Type Annotation）

from typing import TYPE_CHECKING

import torch #PyTorch 是一个用于张量运算、自动微分和深度学习的 Python 框架

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg #Scene Entity Configuration（场景实体配置） 用于定位场景中的某个资产，以及需要时选择该资产中的特定关节、刚体等实体
from isaaclab.utils.math import subtract_frame_transforms #坐标变换函数

if TYPE_CHECKING: #	判断当前代码是否处于静态类型检查语境
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame( #定义一个计算物体相对于机器人根坐标系位置的函数，只返回位置不返回姿态
    env: ManagerBasedRLEnv, #不直接导入，只需要用于类型提示（Type Hint）
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    """robot当前函数中的局部变量；: RigidObject静态类型注解；env强化学习环境对象；.scene环境中的场景管理对象
    robot_cfg机器人实体配置；.name	被选择的场景资产名称；[...]	使用名称访问场景中的资产"""
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3] 
    """.root_pos_w方块根节点在世界坐标系下的位置;[:, :3]取所有环境、每个环境的前三个位置分量"""
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    #作用：根据机器人的世界位置、世界姿态，以及方块的世界位置，计算方块在机器人根坐标系中的位置
    return object_pos_b
