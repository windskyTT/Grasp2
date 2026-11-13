# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.
机器人配置文件
The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
* :obj:`FRANKA_ROBOTIQ_GRIPPER_CFG`: Franka robot with Robotiq_2f_85 gripper

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils # 导入IsaacLab仿真模块（Simulation Module），并使用别名 sim_utils。后续通过它配置 USD 文件、刚体和关节系统的物理属性。
from isaaclab.actuators import ImplicitActuatorCfg # 导入隐式执行器配置类（Implicit Actuator Configuration），用于定义关节驱动的力矩限制、刚度、阻尼等。
from isaaclab.assets.articulation import ArticulationCfg # 	导入多关节机器人配置类（Articulation Configuration），它是整个机器人配置的核心容器。
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# ISAAC_NUCLEUS_DIR	Isaac Sim 机器人及其他资源在 Nucleus 资产库中的根目录常量
# ISAACLAB_NUCLEUS_DIR	Isaac Lab 资源库的根目录常量，用于定位 Isaac Lab 提供的资产

##
# Configuration
## 

# Franka Panda 机械臂 + 原装 Panda 二指夹爪。
# 普通关节位置控制配置，是后面两种配置的基础
FRANKA_PANDA_CFG = ArticulationCfg( #创建一个ArticulationCfg配置对象，将其赋值给变量FRANKA_PANDA_CFG
    spawn=sim_utils.UsdFileCfg( # spawn:加载哪个机器人模型及其物理性质
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False, #不主动为该资产启动lab接触传感器所需的接触报告功能，不等于关闭物理碰撞
        rigid_props=sim_utils.RigidBodyPropertiesCfg( #定义刚体物理属性（Rigid body properties）
            disable_gravity=False, # 不禁用重力
            max_depenetration_velocity=5.0,# 物理引擎用于消除为物体穿透时的最大分离速度，单位m/s
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg( #配置多关节机器人的根关节的物理属性
            enabled_self_collisions=True, #启用机器人自身碰撞
            solver_position_iteration_count=8, #位置约束求解迭代次数
            solver_velocity_iteration_count=0 #速度约束求解迭代次数
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0), #碰撞属性配置
    ),
    init_state=ArticulationCfg.InitialStateCfg( # init_state:机器人初始状态和关节位置
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569, #弧度
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04, #对名称匹配该表达式的夹爪关节设置初始位置 0.04，通常对应 Panda 两个手指各自的移动量，单位为米。
        }, #字典  #`panda_finger_joint.*` 是 正则表达式；.* 表示匹配任意数量的任意字符
    ),
    actuators={ # actuators:关节执行器的控制参数
        "panda_shoulder": ImplicitActuatorCfg( #定义名为 panda_shoulder 的执行器组，使用隐式执行器模型
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0, #设置在仿真物理求解器的力矩上限 N·m
            stiffness=80.0, #PD控制器的P增益(Gain)，也叫刚度
            damping=4.0, #PD控制器的D增益(Gain)，也叫阻尼
        ),
        "panda_forearm": ImplicitActuatorCfg( #前臂执行器
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg( #panda夹抓执行器
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3, #2000
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0, #关节软位置限制范围系数
)
"""Configuration of Franka Emika Panda robot."""

# 与第一种使用相同的机器人模型，但采用更高的 PD 控制增益。
# 主要适合配合逆运动学（Inverse Kinematics，IK）控制。IK控制往往需要高PD
FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True #禁用重力影响，与上面不同
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

# Franka Panda 机械臂 + Robotiq 2F-85 夹爪。
# 需要另外配置 Robotiq 夹爪的关节和执行器。
FRANKA_ROBOTIQ_GRIPPER_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2F_85"}
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0, #内侧手指关节
    ".*_inner_finger_knuckle_joint": 0.0, #内侧手指指节关节
    ".*_outer_.*_joint": 0.0,
}
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.pos = (-0.85, 0, 0.76) #初始位置修改
FRANKA_ROBOTIQ_GRIPPER_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=5200.0,
        velocity_limit_sim=2.175, #仿真关节速度限制（旋转关节rad/s;平移关节:m/s）
        stiffness=1100.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=720.0,
        velocity_limit_sim=2.61,
        stiffness=1000.0,
        damping=80.0,
    ),
    "gripper_drive": ImplicitActuatorCfg( #夹抓主动驱动关节
        joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint（模仿关节/联动关节）
        effort_limit_sim=1650,
        velocity_limit_sim=10.0,
        stiffness=17,
        damping=0.02,
    ),
    # enable the gripper to grasp in a parallel manner 这组参数旨在帮助夹爪以较为平行的方式进行抓持。
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50,
        velocity_limit_sim=10.0,
        stiffness=0.2,
        damping=0.001,
    ),
    # set PD to zero for passive joints in close-loop gripper 针对闭环夹爪中的被动关节设置零 PD 增益。
    "gripper_passive": ImplicitActuatorCfg( #被动关节配置
        joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=10.0,
        stiffness=0.0,
        damping=0.0,
    ),
}


"""Configuration of Franka Emika Panda robot with Robotiq_2f_85 gripper."""

