# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
# 自动发现并导入任务配置包
##

# 从 Isaac Lab 官方任务工具包中导入 import_packages 函数。
from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages 下面配置需要排除的包
_BLACKLIST_PKGS = ["utils", ".mdp"] # 创建黑名单列表，指定自动扫描时跳过名称匹配的包
"""特别提醒：被黑名单跳过，不代表该模块被禁用，也不代表它以后不能导入。"""

# Import all configs in this package 调用自动导入函数
import_packages(__name__, _BLACKLIST_PKGS) #以当前任务包为起点，自动扫描并导入未被排除的子包
