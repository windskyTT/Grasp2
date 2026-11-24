# logs 训练日志

说明：每个标题是该次训练的 checkpoint 目录。“问题”是上一次训练暴露出的问题，即本次修改的动机；没有明确问题的条目省略该节。“修改内容”是相对日志中上一个条目的最终改动。只保留仍有 checkpoint 的 franka_inspire 训练，中间试训已删除。
路径约定：任务代码目录 `source/Grasp2/Grasp2/tasks/manager_based/grasp2/`，下文简称 grasp2；`joint_pos_env_cfg.py` 指 `grasp2/config/franka_inspire/joint_pos_env_cfg.py`。

## logs/rsl_rl/franka_reach/2026-11-01_10-07-40
### 修改内容
- 无修改，Isaac Lab 原始 Franka Reach 任务。

## logs/rsl_rl/franka_lift/2026-11-02_09-24-45
### 修改内容
- 无修改，Isaac Lab 原始 Franka Lift 任务，关节位置控制（joint position control）。

## logs/rsl_rl/franka_lift/2026-11-13_12-17-39
### 修改内容
- 无修改，Isaac Lab 原始 Franka Lift 任务，绝对位姿 IK 控制（IK Abs）。

## logs/rsl_rl/franka_lift/2026-11-14_19-38-59
### 修改内容
- 机器人配置改为 `FRANKA_PANDA_HIGH_PD_CFG`。
- 缩小 `arm_action` 的 scale。
- 新增关节偏离默认位置的惩罚项。

## logs/rsl_rl/franka_inspire/2026-11-18_18-21-48
### 修改内容
将 Franka 夹爪替换为 Inspire RH56DFX 右手，新建 grasp2 任务（lift 任务保持不变）。
- `mdp/rewards.py`
  - 新增函数：`_ee_object_distance`、`fingertips_object_distance`、`hand_closure_near_object`（未启用）、`_finger_object_contact`、`_good_finger_contact`、`good_finger_contact`、`action_rate_l2_clamped`。
  - 修改函数：`object_goal_distance`（增加接触门控参数）。
- 机器人与动作
  - 配置目录 `franka` 改为 `franka_inspire`；环境类改为 `FrankaInspireCubeLiftEnvCfg` / `FrankaInspireCubeLiftEnvCfg_PLAY`；机器人配置改为 `FRANKA_INSPIRE_HIGH_PD_CFG`。
  - 新增 `ARM_INIT_JOINT_POS`、`HAND_INIT_JOINT_POS`，覆盖机械臂与手指初始关节角。
  - `gripper_action` 由 `BinaryJointPositionActionCfg` 改为 `JointPositionActionCfg`，控制 `INSPIRE_ACTIVE_JOINTS` 的 6 个主动关节（scale=0.5，use_default_offset=True）。
- 末端坐标系
  - `object_pose.body_name` 与 `ee_frame` 目标刚体由 `panda_hand` 改为 `rh_hand_base`；末端偏移由 (0, 0, 0.1034) 改为 `GRASP_CENTER_OFFSET` = (-0.054, 0.07, 0.014)。
- 接触传感器
  - 机器人与方块启用 `activate_contact_sensors`。
  - 新增 `arm_contact`、`finger_object_contact` 两个 `ContactSensorCfg`（history_length=3）；新增 `INSPIRE_FINGER_COLLISION_LINKS`，方块上的传感器过滤 33 个手指碰撞连杆。
- 奖励（`lift_env_cfg.py`）
  - 新增 `fingertip_reaching`（权重 2，std=0.05，gate_distance=0.08）、`good_finger_contact`（权重 2，阈值 1 N）、`arm_contact_penalty`（权重 -2，阈值 1 N，panda_link1~7）、`hand_joint_vel`（权重 -1e-4，手指主动关节）、`undesired_contacts`。
  - 两个 `object_goal_tracking` 项增加接触门控：拇指与至少一根其他手指的接触力均 > 1 N 才给分。
  - `action_rate` 改用 `action_rate_l2_clamped`，截断到 [0, 1000]。
- 任务注册与训练配置
  - 任务 ID：`Isaac-Grasp2-Cube-Franka-Inspire-v0` / `Isaac-Grasp2-Cube-Franka-Inspire-Play-v0`，只注册关节位置控制和 RSL-RL（不含 lift 中的 IK、Teddy Bear 及 rl_games/skrl/sb3/robomimic 配置）。
  - PPO `experiment_name` 由 `franka_lift` 改为 `franka_inspire`。
- 方块初始高度 z 为 0.024 m（边长 4.8 cm，贴合桌面；替换前为 0.055 m）。

## logs/rsl_rl/franka_inspire/2026-11-23_11-06-47
### 修改内容
相对 2026-11-18_18-21-48（奖励、动作、观测、课程、终止条件、PPO 超参数均不变）：
- 机器人模型：改为 DemoGrasp 式安装（panda_link8 → L 形转接件 `dg_L_flange` → 手），USD 由 `assets/robots/franka_inspire/franka_inspire.usd` 改为 `assets/robots/franka_inspire_dg/franka_inspire_dg.usd`（`joint_pos_env_cfg.py` 新增 `FRANKA_INSPIRE_DG_USD_PATH`）。
- `ARM_INIT_JOINT_POS`（j1~j7，rad）：(0.378, 0.078, 0.201, -2.087, -1.232, 1.092, -0.104) → (-0.066, -0.166, -0.061, -2.408, -0.017, 1.734, -0.087)，即 DemoGrasp 示范 t=0 的手腕位姿。
- `HAND_INIT_JOINT_POS` 改为预抓取手型：`rh_thumb_1_joint` 0 → 0.596 rad（拇指侧摆到四指对面），四指第 1 关节 0.14 → 0（关节下限，完全张开）；拇指其余关节与四指第 2 关节仍为 0。
- 迭代轮数 1500 → 3001（命令行设置），从头训练。

## logs/rsl_rl/franka_inspire/2026-11-23_13-54-08
### 问题
2026-11-23_11-06-47（model_3000，64 个环境评估一个完整回合）：能稳定抓起方块，但抓取方式与目标位置跟踪存在问题。
- 抬起：64/64 个环境将方块抬离桌面（最低角点 > 1.5 cm），首次离桌中位数第 34 步；第 150 步后 94% 的时间保持离桌，最低角点平均高度约 27 cm。训练曲线 `lifting_object` 11.4/15，`good_finger_contact` 1.58/2，动作噪声 0.76（未收敛）。
- 抓取方式：手抓住方块后手腕翻转为掌心朝上，用手掌托住方块，四指包裹（视频 `videos/play/model_3000.mp4`）。离桌期间手指合力竖直分量为 -3 ~ -9 N（向下），说明方块主要由手掌（不在接触传感器过滤范围内）支撑，而非拇指与四指对向夹持。离桌期间接触最多的连杆为 rh_little_1（96%）、rh_thumb_3（87%）、rh_thumb_rubber_3（68%）；中指仅 11% 的环境参与。抓取过程中方块在手中翻转。
- 抓取前：方块在桌面上且手指向下压 > 2 N 的步数占 10.9%；部分环境在抓起前将方块沿桌面推动。
- 目标跟踪：第 200 步方块到目标距离平均 13.3 cm。方块位置与目标在 y 方向相关系数 0.85，在 x、z 方向仅 0.06、0.02，即只跟随左右方向的目标，前后与高度基本固定（方块平均高度 32 cm，目标平均 39 cm）。训练曲线 `object_goal_tracking_fine_grained` 0.18/5。
- 机械臂姿态：持握时 panda_joint4（肘）均值 -3.07 rad，压在下限 -3.07；panda_joint6（腕俯仰）均值 3.74 rad，压在上限 3.75。手腕翻转到掌心朝上依赖这两个关节折到极限，视频中手臂各关节挤成一团。其余关节离极限均 > 1.5 rad。
### 修改内容
- `mdp/rewards.py`：新增 `joint_limit_margin`，各关节 clamp((margin - 离最近极限的距离) / margin, 0, 1) 之和；离极限 ≥ margin 时为 0，压在极限上时每个关节为 1。
- `lift_env_cfg.py`：新增奖励项 `arm_joint_limit`（`joint_limit_margin`，权重 -2，margin=0.3 rad，关节 `panda_joint.*`）。按 2026-11-23_11-06-47 的持握姿态（两个关节压在极限上）约 -4/步，抬起奖励为 15/步。
- 从 2026-11-23_11-06-47 的 model_3000 继续训练：`--resume --load_run 2026-11-23_11-06-47 --checkpoint model_3000.pt`。
