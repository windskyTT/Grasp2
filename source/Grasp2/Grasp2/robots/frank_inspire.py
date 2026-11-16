# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda 机械臂 + Inspire RH56DFX 右手（灵巧手）的机器人配置。

可用配置:

* :obj:`FRANKA_INSPIRE_CFG`: Franka + Inspire 右手，普通 PD 增益
* :obj:`FRANKA_INSPIRE_HIGH_PD_CFG`: 机械臂使用更高 PD 增益并关闭重力（适合关节位置控制/IK 控制）

模型来源:
    assets/robots/franka/franka_inspire/franka_inspire.usd
    由 assets/robots/franka/build_franka_inspire_urdf.py + convert_franka_inspire_usd.py 生成。

关节说明（共 19 个，可通过 robot.joint_names 查看）:
    机械臂 7 个:      panda_joint1 ~ panda_joint7
    手主动关节 6 个:  rh_thumb_1_joint   拇指侧摆/对掌   [0, 1.310]
                      rh_thumb_2_joint   拇指弯曲        [0, 0.523]
                      rh_index_1_joint   食指            [0, 1.344]
                      rh_middle_1_joint  中指            [0, 1.344]
                      rh_ring_1_joint    无名指          [0, 1.344]
                      rh_little_1_joint  小指            [0, 1.344]
    手 mimic 关节 6 个（PhysX Mimic Joint 自动跟随，不要配置执行器、不要下指令）:
                      rh_{index,middle,ring,little}_2_joint = 1.1169 * *_1_joint - 0.15
                      rh_thumb_3_joint = 1.1425 * rh_thumb_2_joint
                      rh_thumb_4_joint = 0.7508 * rh_thumb_3_joint
    腕部的 rh_wrist_yaw_joint / rh_hand_base_joint 在 URDF 中已改为固定关节。

常用刚体名称:
    rh_hand_base（手掌基座，手指沿其 +Y 方向伸出）、rh_palm、rh_{thumb,index,middle,ring,little}_tip
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 项目根目录: .../Grasp2/source/Grasp2/Grasp2/robots/frank_inspire.py -> parents[4] = .../Grasp2
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
FRANKA_INSPIRE_USD_PATH = str(_PROJECT_ROOT / "assets/robots/franka/franka_inspire/franka_inspire.usd")

# 灵巧手关节名（正则）
INSPIRE_ACTIVE_JOINTS = ["rh_thumb_[12]_joint", "rh_(index|middle|ring|little)_1_joint"]
"""6 个主动（有电机）的手指关节。"""
INSPIRE_MIMIC_JOINTS = ["rh_thumb_[34]_joint", "rh_(index|middle|ring|little)_2_joint"]
"""6 个 mimic 被动关节。"""

# 四指 *_1 关节在 0.1343 rad 时，mimic 的 *_2 关节正好为 0。
# 小于这个值时远端指节会反向弯曲（URDF 原始模型的问题），因此“张开”状态取 0.14。
INSPIRE_FINGER_OPEN = 0.14
INSPIRE_THUMB_OPPOSE = 1.1
"""拇指侧摆到对掌位置（rh_thumb_1_joint），抓取时通常保持不变。"""


##
# Configuration
##

FRANKA_INSPIRE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRANKA_INSPIRE_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 手指之间（尤其拇指对掌时与手掌）距离很近，开启自碰撞容易互相卡住/抖动，先关闭。
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # 机械臂: 与 FRANKA_PANDA_CFG 相同的初始姿态
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            # 灵巧手: 拇指对掌、四指张开
            "rh_thumb_1_joint": INSPIRE_THUMB_OPPOSE,
            "rh_thumb_[234]_joint": 0.0,
            "rh_(index|middle|ring|little)_1_joint": INSPIRE_FINGER_OPEN,
            "rh_(index|middle|ring|little)_2_joint": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # 只驱动 6 个主动关节；mimic 关节由 PhysX 约束带动，不放进任何执行器组
        # （IsaacLab 会提示 "Not all actuators are configured"，属正常现象）。
        "inspire_hand": ImplicitActuatorCfg(
            joint_names_expr=INSPIRE_ACTIVE_JOINTS,
            effort_limit_sim=5.0,  # N·m，URDF 标称 10；先取较小值，抓不住再加大
            velocity_limit_sim=2.0,  # rad/s，与 URDF 一致
            stiffness=20.0,
            damping=1.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Franka Panda + Inspire RH56DFX 右手。"""


FRANKA_INSPIRE_HIGH_PD_CFG = FRANKA_INSPIRE_CFG.copy()
FRANKA_INSPIRE_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_INSPIRE_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_INSPIRE_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_INSPIRE_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_INSPIRE_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Franka Panda + Inspire 右手，机械臂使用更高 PD 增益（与 FRANKA_PANDA_HIGH_PD_CFG 对应）。"""
