# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
argparse 是 Python 标准库提供的命令行参数解析模块（Argument Parsing Module）
它能够把终端输入转换成程序可以使用的 Python 变量。
| 名称 | 中文含义 | 功能 |
|--------|--------|--------|
| `ArgumentParser` | 参数解析器 | 定义允许使用哪些命令行参数 |
| `add_argument()` | 添加参数 | 注册一个具体选项 |
| `parse_known_args()` | 解析已知参数 | 解析已注册的选项，并保留不认识的参数 |
| `Namespace` | 命名空间对象 | 保存解析结果，使其可以通过属性访问 |
"""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):#向命令行解析器注册 RSL-RL 参数
    #函数接收已经存在的解析器，并在这个解析器上增加参数，而不是重新创建一个解析器
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group 创建参数分组
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments experiment_name:区分不同实验集合  run_name:区分同一实验下的不同训练运行
    arg_group.add_argument( #将选项注册到 RSL-RL 参数组       
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    ) #default=None未指定时保持为 None
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.") #恢复训练
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.") #指定之前训练的运行目录
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    ) #tensorboard 	TensorBoard 本地查看损失、奖励、学习率等  wandb	Weights & Biases在线实验管理与可视化 neptune Neptune 实验数据管理与跟踪
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg: #	加载并解析RSL-RL配置
    """读取指定任务的默认配置，并将命令行参数应用到配置中
    Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments. 命令行参数

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    """load_cfg_from_registry() 的含义：从 Gymnasium 任务注册表（Gymnasium Registry）中找到指定任务对应的配置入口，并加载配置"""

    # load the default configuration 记载默认配置
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace): #更新已有训练配置
    """函数的核心作用：检查哪些命令行参数需要覆盖原始配置，并直接修改配置对象中的对应属性
    Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint #命令行参数叫 checkpoint，而配置对象属性叫 load_checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
