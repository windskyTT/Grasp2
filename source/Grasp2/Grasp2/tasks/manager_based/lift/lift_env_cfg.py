# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Lift强化学习环境配置文件
from dataclasses import MISSING # 数据类（data class），missing是特殊的标记对象，用于表示某项配置尚未提供有效值

import isaaclab.sim as sim_utils # 导入仿真相关配置模块，使用 sim_utils 别名
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm # 课程学习项配置
from isaaclab.managers import EventTermCfg as EventTerm #事件项配置
from isaaclab.managers import ObservationGroupCfg as ObsGroup # 观测组配置
from isaaclab.managers import ObservationTermCfg as ObsTerm # 单个观测项配置
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg # 交互式仿真场景的基础配置类
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg # 坐标系转换传感器配置
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass # lab配置类装饰器
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR # sim资产目录常量

from . import mdp

##
# Scene definition
##


@configclass # 装饰器用于为类或函数增加特定行为
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg 机器人将在具体环境配置中填入
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg 末端坐标传感器
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg 刚体或可变性物体
    object: RigidObjectCfg | DeformableObjectCfg = MISSING 

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    # 地面是使用绝对 USD 路径定义的共享场景元素
    # 而桌子使用环境命名空间进行配置
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:# 目标命令配置
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg( # 创建名为 object_pose 的均匀随机位姿命令配置 
        #  `object_pose` 是一个命令项名称，用来提供任务希望追踪的目标位姿
        asset_name="robot", # 指定该命令关联的场景资产名称为 robot
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0), # 目标命令的重采样时间间隔 5 秒
        debug_vis=True, # 开启命令生成器的调试可视化
        ranges=mdp.UniformPoseCommandCfg.Ranges( # 指定随机目标位姿的取值范围
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    # 	声明机械臂动作接口，允许使用关节位置控制或者微分逆运动学控制
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING # 声明夹爪动作接口，采用二值关节位置动作配置类型


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup): # 定义策略观测组，继承ObserationGroupCfg
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame) #将物体相对机器人根坐标系的位置加入观测
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action) # 将上一个时间步的动作加入观测

        def __post_init__(self): #定义观测组配置初始化完成后的处理函数
            self.enable_corruption = True # 启用观测扰动处理机制 Corruption（扰动／观测噪声处理）设置 enable_corruption=True 不等于一定为所有观测添加了非零随机噪声
            self.concatenate_terms = True # 将多个观测项拼接成一个观测张量

    # observation groups
    policy: PolicyCfg = PolicyCfg() #创建名为policy的观测组配置


@configclass
class EventCfg: #定义事件配置类
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset") #创建名为 reset_all 的重置事件，调用 reset_scene_to_default

    reset_object_position = EventTerm( # 创建一个用于随机重置物体状态的事件
        func=mdp.reset_root_state_uniform, # 均匀随机重置物体根状态的函数
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)}, #设置物体位置与姿态的随机偏移范围
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"), #指定操作对象是场景中名称为 object 的资产
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0) #接近物体奖励

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0) # 物体抬升奖励

    object_goal_tracking = RewTerm( #定义目标位置追踪奖励
        func=mdp.object_goal_distance, # 使用物体到目标位置距离的奖励函数
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm( #定义精细目标追踪奖励
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty 动作惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm( #定义关节速度惩罚
        func=mdp.joint_vel_l2, # 使用关节速度平方和函数
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")}, # 设定对象为机器人
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True) # 创建超时终止项，标记为时间限制类型

    object_dropping = DoneTerm( # 物体掉落终止项
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    ) # 当物体根节点高度低于 -0.05 m 时触发相应终止条件


@configclass
class CurriculumCfg: #课程学习配置类 Curriculum Learning（课程学习） 是一种通过调整训练任务、训练数据或优化条件，让智能体逐步学习的机制
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm( # 定义动作变化惩罚权重调整项
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm( # 定义关节速度惩罚权重调整项
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5) #环境间距2.5M
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2 # 每个强化学习环境控制步执行 2 次物理仿真步
        self.episode_length_s = 5.0 # 设置每回合最大仿真时长为 5 秒
        # simulation settings
        self.sim.dt = 0.01  # 100Hz 设置物理仿真时间步为 0.01 秒，即 100 Hz
        self.sim.render_interval = self.decimation # 将渲染间隔设置为 2 个物理仿真步

        #Bounce Threshold Velocity（反弹速度阈值）是PhysX接触求解中的一个参数。影响碰撞接触在什么相对速度条件下考虑反弹响应
        self.sim.physx.bounce_threshold_velocity = 0.2 # 暂时将 PhysX 的碰撞反弹速度阈值配置为 0.2 m/s
        self.sim.physx.bounce_threshold_velocity = 0.01 # 再次赋值为 0.01 m/s，覆盖上一行
        # GPU碰撞资源容量
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4 # 设置 GPU 碰撞相关的 found/lost aggregate pairs 容量为 4,194,304
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 # 设置 GPU aggregate pairs 总容量为 16,384
        self.sim.physx.friction_correlation_distance = 0.00625 # 设置摩擦接触关联距离参数为 0.00625 m
