# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms #导入坐标变换组合函数，用于将基座坐标系中的目标位置转换到世界坐标系

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted( #物体抬升奖励
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance( # Franka Panda 末端执行器距离方块有多远，并将距离转换为奖励
    env: ManagerBasedRLEnv,
    std: float, #距离奖励的尺度参数
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.采用 tanh 核函数计算接近奖励"""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3) 目标物体张量形状
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :] #（并行环境数，被FrameTransformer跟踪的目标坐标系数量，三维坐标）
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1) #计算方块与末端执行器之间的欧几里得距离

    return 1 - torch.tanh(object_ee_distance / std) #将距离映射为 0～1 之间的连续奖励


def object_goal_distance( #目标位置跟踪奖励
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel.

    contact_sensor_cfg: 若给定（手指-物体接触传感器），还要求拇指+另一根手指接触物体才给奖励
        （仿照 Dexsuite 的 position_command_error_tanh），防止把物体拍飞/弹起也拿分。
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b) # 将目标位置从机器人根坐标系转换为世界坐标系
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    reward = (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
    if contact_sensor_cfg is not None:
        reward = reward * good_finger_contact(env, threshold=contact_threshold, sensor_cfg=contact_sensor_cfg)
    return reward


##
# 灵巧手抓取奖励（Inspire RH56DFX）
##


def _ee_object_distance(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """抓握中心（ee_frame 的第一个目标坐标系）到物体中心的距离: (num_envs,)"""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)


def fingertips_object_distance(  # 指尖接近物体奖励
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    gate_distance: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """各指尖到物体中心的平均距离，用 tanh 核映射到 (0, 1]。

    Args:
        std: 距离尺度 (m)。
        robot_cfg: 机器人实体配置，必须在 params 中显式给出 body_names（指尖刚体），
            这样管理器才会解析出 body_ids。
        gate_distance: 若给定，只有抓握中心到物体的距离小于该值时才给奖励，
            避免手掌还没到位时手指就提前伸向物体。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]  # (num_envs, num_tips, 3)
    dist = torch.norm(tips_w - object.data.root_pos_w[:, None, :], dim=-1).mean(dim=-1)  # (num_envs,)
    reward = 1.0 - torch.tanh(dist / std)
    if gate_distance is not None:
        reward = reward * (_ee_object_distance(env, object_cfg, ee_frame_cfg) < gate_distance)
    return reward


def hand_closure_near_object(  # 物体在手掌下方时的握拳奖励
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    contact_sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """抓握中心距物体小于 threshold 时，奖励手指关节的闭合程度。

    闭合程度 = 各主动手指关节在其软限位范围内的归一化位置的平均值（0 = 完全张开，1 = 完全闭合）。

    Args:
        threshold: 抓握中心到物体中心的距离阈值 (m)。
        asset_cfg: 机器人实体配置，必须在 params 中显式给出 joint_names（主动手指关节）。
        contact_sensor_cfg: 若给定（手指-物体接触传感器），还要求拇指+另一根手指真正接触物体，
            否则不给奖励。避免“手掌压在物体上、手指空握到关节极限”也能拿满分。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]  # (num_envs, num_joints, 2)
    closure = ((q - limits[..., 0]) / (limits[..., 1] - limits[..., 0])).clamp(0.0, 1.0).mean(dim=-1)
    near = _ee_object_distance(env, object_cfg, ee_frame_cfg) < threshold
    reward = closure * near
    if contact_sensor_cfg is not None:
        reward = reward * good_finger_contact(env, threshold=contact_threshold, sensor_cfg=contact_sensor_cfg)
    return reward


##
# 手指-物体接触奖励（仿照 Isaac-Dexsuite-Kuka-Allegro-Lift-v0 的 good_finger_contact）
##


def _finger_object_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float #接触力阈值
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (拇指是否接触物体, 其余四指中接触物体的手指数)，形状均为 (num_envs,)。

    接触传感器装在物体上（单个刚体），filter_prim_paths_expr 逐个列出手指连杆（每条只匹配一个连杆）。
    Isaac Lab 的过滤接触只支持“一个传感器刚体 对 多个过滤刚体”，所以传感器必须放在物体上，
    而不是放在多个手指连杆上（那样 force_matrix_w 恒为 0）。
    force_matrix_w: (num_envs, 1, num_filters, 3)，第 k 列对应 filter_prim_paths_expr[k] 的连杆。
    某根手指上任意一个连杆的接触力 > threshold 即认为该手指接触了物体。
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] 
    force = sensor.data.force_matrix_w[:, 0].norm(dim=-1)  # (num_envs, num_filters) 从接触力矩阵中获取物体与每个被过滤手指连杆的接触力，并计算各个接触力向量的大小
    touch = force > threshold
    names = [expr.rsplit("/", 1)[-1] for expr in sensor.cfg.filter_prim_paths_expr]
    thumb = touch[:, [i for i, n in enumerate(names) if n.startswith("rh_thumb_")]].any(dim=-1)
    others = torch.zeros_like(thumb, dtype=torch.long)
    for finger in ("index", "middle", "ring", "little"):
        ids = [i for i, n in enumerate(names) if n.startswith(f"rh_{finger}_")]
        others += touch[:, ids].any(dim=-1).long() #any() 表示只要指定维度内至少有一个元素为 True，结果就为 True long()将布尔值转换成整数
    return thumb, others #拇指是否与物体接触；其余四根手指中，有多少根与物体接触


def good_finger_contact(  # 有效抓握接触奖励
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("finger_object_contact"),
) -> torch.Tensor:
    """拇指与至少一根其他手指同时接触物体（接触力 > threshold N）时为 1，否则为 0: (num_envs,) float。

    既作为奖励项使用（RewTerm），也被 object_goal_distance、hand_closure_near_object 用作接触门控。
    """
    thumb, others = _finger_object_contact(env, sensor_cfg, threshold)
    return (thumb & (others > 0)).float()


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """动作变化率 L2 惩罚，并截断到 [0, 1000]（仿照 Dexsuite），防止动作突变时惩罚爆炸。"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(0.0, 1000.0)


def joint_limit_margin(  # 关节靠近极限惩罚（权重取负）
    env: ManagerBasedRLEnv,
    margin: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """各关节离最近关节极限不足 margin (rad) 的程度之和: (num_envs,)。

    每个关节的取值 = clamp((margin - 离极限距离) / margin, 0, 1)：离极限 >= margin 时为 0，压在极限上时为 1。
    用于防止机械臂为了翻转手腕把肘、腕关节折到极限、手臂挤成一团。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lim = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    dist = torch.minimum(q - lim[..., 0], lim[..., 1] - q)
    return ((margin - dist) / margin).clamp(0.0, 1.0).sum(dim=1)
