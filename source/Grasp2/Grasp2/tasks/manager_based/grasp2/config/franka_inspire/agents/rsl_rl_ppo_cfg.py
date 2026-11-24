# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
用来告诉 RSL-RL：
- 使用什么神经网络。
- 每次收集多少步训练数据。
- 使用多少轮训练迭代。
- 学习率是多少。
- 怎样控制 PPO 更新幅度。
- 多久保存一次训练模型。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
                                # 用途：组织整个训练流程       设置神经网络结构        设置优化算法超参数
"""
`RslRlOnPolicyRunnerCfg`
Runner（运行器）用于协调训练过程。
它负责的工作通常包括：与环境交互、收集数据、调用算法更新、记录训练指标和保存模型。
On-Policy（同策略）表示 PPO 主要使用由当前或最近一次采样策略生成的数据进行更新。
它不会像典型的 Off-Policy（离策略）算法那样长期反复使用大型历史经验回放池。

`RslRlPpoActorCriticCfg`
它用于配置 Actor-Critic（演员—评论家）网络结构。
Actor 负责生成动作，Critic 负责评价状态的价值。

`RslRlPpoAlgorithmCfg`
它用于配置 PPO 算法更新方式，包括学习率、折扣因子、优势估计、裁剪范围等
"""
@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 配置采样步数、迭代次数、保存间隔
    num_steps_per_env = 24 #每个环境在一次采样阶段收集的步数
    max_iterations = 1500
    save_interval = 50 #模型保存间隔
    experiment_name = "franka_inspire" #实验名称 用于组织训练日志和模型保存目录
    # policy 决定机器人用什么模型作出决策，`algorithm` 决定怎样训练这个模型
    policy = RslRlPpoActorCriticCfg( # 配置 Actor-Critic 神经网络 定义神经网络结构（Netwwork Architecture)
        init_noise_std=1.0, #初始动作分布标准差1.0 采用高斯分布
        actor_obs_normalization=False, #Actor观测归一化
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64], #三层隐藏层 神经元数量分别为256、128、64 输出8维动作均值
        critic_hidden_dims=[256, 128, 64], #输出1维状态均值
        activation="elu", # 指数线性单元 elu激活函数
    )
    algorithm = RslRlPpoAlgorithmCfg( # 配置 PPO 算法超参数 定义神经网络如何学习(Optimization)
        value_loss_coef=1.0, #价值损失权重 Value Loss Coefficient
        use_clipped_value_loss=True, # Clipped Value Function Loss 裁剪价值函数损失
        clip_param=0.2, #PPO Clipping Parameter PPO 裁剪参数
        entropy_coef=0.006, #熵系数
        num_learning_epochs=5, #每次 PPO 迭代中的学习轮数
        num_mini_batches=4, #小批次数量
        learning_rate=1.0e-4,
        schedule="adaptive", #自适应学习率调度 Adaptive Learning Rate Schedule 当前 RSL-RL 实现会观察新旧策略的 KL 散度
        gamma=0.98, #折扣因子
        lam=0.95, #广义优势估计参数 Generalized Advantage Estimation（广义优势估计）
        desired_kl=0.01, #目标KL散度
        max_grad_norm=1.0, #最大梯度范数
    )
