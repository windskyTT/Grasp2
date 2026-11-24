# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Franka + Inspire RH56DFX 灵巧手 的 Grasp2 抓取环境配置（关节位置控制）
# 在通用 LiftEnvCfg 的基础上：
#   - 机器人换成 FRANKA_INSPIRE_HIGH_PD_CFG，USD 换成 DemoGrasp 式安装（L 形转接件）
#   - 机械臂 7 个关节 + 灵巧手 6 个主动关节，全部使用关节位置控制
#   - 末端坐标系 ee_frame 放在手掌正下方（抓握中心）

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ... import mdp
from ...lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from Grasp2.robots.frank_inspire import (  # isort: skip
    FRANKA_INSPIRE_HIGH_PD_CFG,
    FRANKA_INSPIRE_USD_PATH,
    INSPIRE_ACTIVE_JOINTS,
)

# 机器人 USD：DemoGrasp 式安装（panda_link8 -> L 形转接件 dg_L_flange -> 手），
# 由 assets/robots/build_franka_inspire_urdf.py --mount demograsp 生成。
FRANKA_INSPIRE_DG_USD_PATH = FRANKA_INSPIRE_USD_PATH.replace("franka_inspire/franka_inspire.usd", "franka_inspire_dg/franka_inspire_dg.usd")

# 机械臂初始关节角：DemoGrasp 示范 t=0 的手腕位姿
ARM_INIT_JOINT_POS = {
    "panda_joint1": -0.066,
    "panda_joint2": -0.166,
    "panda_joint3": -0.061,
    "panda_joint4": -2.408,
    "panda_joint5": -0.017,
    "panda_joint6": 1.734,
    "panda_joint7": -0.087,
}
# 灵巧手初始状态：参考抓取的预抓取手型——拇指侧摆 0.596 rad 转到四指对面并伸直，四指完全张开（关节下限 0）。
HAND_INIT_JOINT_POS = {
    "rh_thumb_1_joint": 0.596,
    "rh_thumb_2_joint": 0.0,
    "rh_thumb_3_joint": 0.0,
    "rh_thumb_4_joint": 0.0,
    "rh_(index|middle|ring|little)_1_joint": 0.0,
    "rh_(index|middle|ring|little)_2_joint": 0.0,
}
# 五根手指上所有带碰撞体的连杆（不含手掌），用于手指-物体接触检测
INSPIRE_FINGER_COLLISION_LINKS = [
    "rh_thumb_1", "rh_thumb_2", "rh_thumb_rubber_1", "rh_thumb_3", "rh_thumb_rubber_2",
    "rh_thumb_4", "rh_thumb_rubber_3", "rh_thumb_pad", "rh_thumb_tip",
] + [
    f"rh_{finger}_{part}"
    for finger in ("index", "middle", "ring", "little")
    for part in ("1", "rubber_1", "2", "rubber_2", "pad", "tip")
]

# 抓握中心在 rh_hand_base 坐标系下的位置：手掌中心正下方 2.7 cm（rh_hand_base 的 -X 方向为手掌法向，手指沿 +Y）
GRASP_CENTER_OFFSET = (-0.054, 0.07, 0.014)


@configclass
class FrankaInspireCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 机器人：Franka + Inspire 右手，覆盖初始关节角（use_default_offset 会以此为动作零点）
        # 打开接触力上报（在 spawn 的副本上设置，不改 frank_inspire.py），供 arm_contact 传感器使用
        self.scene.robot = FRANKA_INSPIRE_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=FRANKA_INSPIRE_HIGH_PD_CFG.spawn.replace(
                usd_path=FRANKA_INSPIRE_DG_USD_PATH,
                activate_contact_sensors=True,
            ),
        )
        self.scene.robot.init_state.joint_pos = {**ARM_INIT_JOINT_POS, **HAND_INIT_JOINT_POS}

        # 机械臂连杆接触传感器：只覆盖 panda_link*（不含灵巧手），用于惩罚手以外的部位碰桌子/方块。
        # 不使用 filter_prim_paths_expr：Isaac Lab 的过滤接触只支持“一个传感器刚体 对 多个过滤刚体”，
        # 这里传感器覆盖多个连杆，过滤会恒为 0。因此直接用连杆受到的总接触力。
        self.scene.arm_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link.*",
            history_length=3,
        )

        # 机械臂：关节位置控制
        # 目标角 = 默认角 + scale * action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale={"panda_joint[1357]": 0.1, "panda_joint[246]": 0.3},
            use_default_offset=True,
        )
        # 灵巧手：6 个主动关节的关节位置控制（mimic 关节由 PhysX 自动跟随，不在动作空间内）
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=INSPIRE_ACTIVE_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )

        # 目标位姿命令关联到手掌基座
        self.commands.object_pose.body_name = "rh_hand_base"

        # 物体：与 Lift 相同的 DexCube，0.8 倍（边长 4.8 cm，约 0.11 kg）。初始高度 = 半个边长 0.024 m。
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.024], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                activate_contact_sensors=True,  # 供 finger_object_contact 传感器使用
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # 手指-物体接触传感器：传感器装在方块上（单个刚体），逐个过滤 33 个手指连杆。
        # force_matrix_w[:, 0, k] = 方块与 INSPIRE_FINGER_COLLISION_LINKS[k] 之间的接触力。
        # 供 good_finger_contact、object_goal_tracking（接触门控）使用。
        self.scene.finger_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{link}" for link in INSPIRE_FINGER_COLLISION_LINKS],
            history_length=3,
        )

        # 末端坐标系传感器：手掌正下方的抓握中心
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/rh_hand_base",
                    name="end_effector",
                    offset=OffsetCfg(pos=GRASP_CENTER_OFFSET),
                ),
            ],
        )


@configclass
class FrankaInspireCubeLiftEnvCfg_PLAY(FrankaInspireCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # play 时不显示坐标系：关闭目标位姿命令的可视化（目标坐标系 + 手掌基座坐标系）和末端坐标系
        self.commands.object_pose.debug_vis = False
        self.scene.ee_frame.debug_vis = False
