# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Franka 专用的 Lift 环境配置文件（Franka-Specific Lifting Environment Configuration）
# 核心工作是：
# 在通用 LiftEnvCfg 的基础上，指定具体使用 Franka Panda 机器人、Panda 二指夹爪、DexCube 方块和关节位置控制方法

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg # 导入坐标系偏移配置类
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg # 导入刚体物理属性配置类
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip 导入用于显示坐标系的可视化标记配置
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent 需要执行父类的初始化后处理
        super().__post_init__() # 调用父类 LiftEnvCfg 的初始化后处理函数

        # Set Franka as robot 将 Franka 设置为场景机器人
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # PPO 输出的动作，究竟如何转换成 Franka 机械臂和二指夹爪的运动指令？
        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg( # 将机械臂动作配置设置为关节位置动作类型
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True #使用机器人默认关节位置作为动作偏移量
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04}, #指定打开夹爪时的目标手指关节位置
            close_command_expr={"panda_finger_.*": 0.0}, #指定关闭夹爪时的目标手指关节位置
        )

        # Set the body name for the end effector 将目标位姿命令关联到 Franka 机器人的 panda_hand 刚体
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object", #指定方块在各并行环境中的 USD Prim 路径
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg( #配置物理属性
                    solver_position_iteration_count=16, #设置刚体位置约束求解迭代次数为 16
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0, #rad/s
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0, # 设置物理引擎消除物体穿透时的最大分离速度
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms 末端执行器坐标系传感器
        marker_cfg = FRAME_MARKER_CFG.copy() # 复制一份默认坐标轴可视化配置
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1) # 设置坐标轴可视化标记的缩放尺寸
        marker_cfg.prim_path = "/Visuals/FrameTransformer" # 指定调试可视化标记的USD路径
        self.scene.ee_frame = FrameTransformerCfg( # 定义一个名为 ee_frame 的坐标系变换传感器配置
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0", #指定参考在坐标系来源panda_link0
            debug_vis=False, #关闭该传感器的调试可视化
            visualizer_cfg=marker_cfg, # 指定传感器可视化时使用的标记配置
            target_frames=[ #开始定义需要追踪的目标坐标系列表
                FrameTransformerCfg.FrameCfg(  #创建一个目标坐标系配置项
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play 禁用policy观测组的噪声处理开关
        self.observations.policy.enable_corruption = False
