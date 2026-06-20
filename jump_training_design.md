# 跳跃训练方案设计文档

## 1. 观测空间扩展

当前 46 维单帧观测 + 1 维跳跃指令 = 47 维单帧
堆叠 3 帧 = 141 维 actor 观测

新增 obs[46] = jump_command (0=行走, 1=跳跃)

## 2. 跳跃奖励函数

### _reward_jump_height (跳跃高度奖励)
```python
def _reward_jump_height(self):
    """奖励跳跃时的最大高度"""
    jump_cmd = self.commands[:, 5] > 0.5  # 跳跃指令
    all_feet_air = torch.all(
        self.contact_forces[:, self.feet_indices, 2] < 1., dim=1
    )
    height_bonus = torch.clamp(self.root_states[:, 2] - 0.25, min=0)  # 超过25cm的部分
    return jump_cmd.float() * all_feet_air.float() * height_bonus
```
权重: 10.0

### _reward_jump_clearance (越障奖励)
```python
def _reward_jump_clearance(self):
    """奖励成功越过障碍物"""
    # 需要在环境中放置障碍物并检测是否越过
    pass
```
权重: 5.0

### _reward_landing_stability (稳定落地奖励)
```python
def _reward_landing_stability(self):
    """惩罚落地时的冲击力"""
    landing_force = torch.sum(
        self.contact_forces[:, self.feet_indices, 2], dim=1
    )
    # 落地时接触力超过阈值就惩罚
    return torch.clamp(landing_force - 200, min=0)
```
权重: -0.01

### _reward_tuck_legs (收腿奖励)
```python
def _reward_tuck_legs(self):
    """腾空时奖励收腿（膝关节弯曲）"""
    all_feet_air = torch.all(
        self.contact_forces[:, self.feet_indices, 2] < 1., dim=1
    )
    # calf 关节越弯曲（越负）越好
    calf_indices = [2, 5, 8, 11]  # FL/FR/RL/RR calf
    calf_tuck = torch.sum(torch.abs(self.dof_pos[:, calf_indices] - (-2.5)), dim=1)
    return all_feet_air.float() * (1.0 / (1.0 + calf_tuck))
```
权重: 2.0

## 3. 训练 Curriculum

### 阶段 1: 平地跳跃 (iter 0-2000)
- 地形: 平地
- 指令: 50% 行走 (jump=0), 50% 跳跃 (jump=1)
- 行走时使用现有奖励
- 跳跃时激活 jump_height + tuck_legs + landing_stability
- 目标: 学会原地起跳和落地

### 阶段 2: 低障碍越跳 (iter 2000-4000)
- 地形: 随机放置 5-8cm 障碍物
- 指令: 前进 + 跳跃
- 新增 jump_clearance 奖励
- 目标: 学会跑到障碍物前起跳越过

### 阶段 3: 高障碍越跳 (iter 4000-6000)
- 地形: 随机放置 8-15cm 障碍物
- 指令: 前进 + 自动跳跃（策略自主判断）
- 可选: 移除 jump 指令，让策略从地形观测自动决定
- 目标: 自主判断何时跳跃

## 4. 关键参数

### 指令空间
```python
num_commands = 6  # vx, vy, yaw, heading, height, jump
jump 范围: {0, 1} (二值)
```

### 奖励权重
```python
# 行走奖励 (jump=0 时)
tracking_lin_vel = 1.5
tracking_ang_vel = 0.5
feet_air_time = 10.0
orientation = -5.0

# 跳跃奖励 (jump=1 时)  
jump_height = 10.0
tuck_legs = 2.0
landing_stability = -0.01

# 通用奖励 (始终生效)
action_rate = -0.05
torques = -0.0002
dof_pos_limits = -5.0
```

### 训练参数
- 从当前最佳策略 (12_bc_highspeed_mujoco.pt) 开始
- Isaac Gym 4096 环境
- 学习率 1e-3 (adaptive)
- 先平地训练跳跃，再加障碍物

## 5. 注意事项

1. 跳跃时 action_scale 可能需要加大 (0.25 → 0.5) 以允许更大的关节运动
2. 起跳瞬间需要所有腿同时发力，PD 增益可能需要提高
3. 落地缓冲需要较软的 PD 响应，考虑动态调节 kp/kd
4. 跳跃和行走的切换需要平滑过渡
5. 真机部署时需要额外的安全机制（力矩限制、姿态保护）
