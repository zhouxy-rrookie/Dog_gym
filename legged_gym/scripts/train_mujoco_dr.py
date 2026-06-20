"""MuJoCo PPO training with domain randomization for sim-to-real transfer.
Fine-tunes from MuJoCo-adapted policy, adding latency/gain/noise DR."""

import os, sys, time, copy
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, '/root/gpufree-data/workspace/legged_gym')
sys.path.insert(0, '/root/gpufree-data/workspace/rsl_rl')

import mujoco
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

MJCF_PATH = '/root/gpufree-data/workspace/legged_gym/resources/robots/dog_urdf/urdf/dog_with_meshes.mjcf'
PRETRAINED = '/root/gpufree-data/workspace/legged_gym/weights/03_mujoco_finetuned.pt'
SAVE_DIR = '/root/gpufree-data/workspace/legged_gym/logs/mujoco_dr_latency'

NUM_ENVS = 16
SIM_DT = 0.002
DECIMATION = 10
POLICY_DT = SIM_DT * DECIMATION
EPISODE_LENGTH = int(20.0 / POLICY_DT)
ACTION_SCALE = 0.25
NUM_OBS = 49
NUM_ACTIONS = 12

DEFAULT_ANGLES = np.array([0.1,0.8,-1.5,0.1,0.8,-1.5,0.1,1.0,-1.5,0.1,1.0,-1.5], dtype=np.float32)

REWARD_SCALES = {
    'tracking_lin_vel': 1.5, 'tracking_ang_vel': 0.5,
    'lin_vel_z': -2.0, 'ang_vel_xy': -0.05,
    'orientation': -1.0, 'base_height': -1.0,
    'torques': -0.0002, 'dof_pos_limits': -5.0,
    'dof_acc': -2.5e-7, 'action_rate': -0.05,
    'feet_air_time': 10.0, 'foot_slip': -0.1,
}
for k in REWARD_SCALES:
    REWARD_SCALES[k] *= POLICY_DT

TRACKING_SIGMA = 0.25
BASE_HEIGHT_TARGET = 0.25
SOFT_DOF_POS_LIMIT = 0.9
CMD_RANGES = {'vx': [-1,1], 'vy': [-0.5,0.5], 'vyaw': [-1,1], 'heading': [-3.14,3.14], 'height': [0.20, 0.30]}
OBS_SCALES = {'lin_vel': 2.0, 'ang_vel': 0.25, 'dof_pos': 1.0, 'dof_vel': 0.05}
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

DR = {
    'obs_latency_max': 0,
    'action_latency_max': 1,
    'kp_range': [20.0, 20.0],
    'kd_range': [0.5, 0.5],
    'motor_offset_range': [0.0, 0.0],
    'push_interval': 999999,
    'push_vel_range': [0.0, 0.0],
    'obs_noise': {
        'lin_vel': 0.0, 'ang_vel': 0.0, 'dof_pos': 0.0, 'dof_vel': 0.0, 'gravity': 0.0,
    },
}


class MuJoCoVecEnv:
    def __init__(self, num_envs):
        self.n = num_envs
        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.model.opt.timestep = SIM_DT
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]

        self.foot_body_ids = []
        self.thigh_body_ids = []
        self.calf_body_ids = []
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        for name in ['FL_foot','FR_foot','RL_foot','RR_foot']:
            self.foot_body_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in ['FL_thigh','FR_thigh','RL_thigh','RR_thigh']:
            self.thigh_body_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in ['FL_calf','FR_calf','RL_calf','RR_calf']:
            self.calf_body_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))

        m = self.model
        self.dof_pos_limits = np.zeros((12, 2), dtype=np.float32)
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and 'joint' in name and name != 'root':
                idx = m.jnt_qposadr[j] - 7
                r = m.jnt_range[j]
                mid = (r[0] + r[1]) / 2
                half = (r[1] - r[0]) / 2
                self.dof_pos_limits[idx, 0] = mid - half * SOFT_DOF_POS_LIMIT
                self.dof_pos_limits[idx, 1] = mid + half * SOFT_DOF_POS_LIMIT

        self.obs = np.zeros((num_envs, NUM_OBS), dtype=np.float32)
        self.actions = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float32)
        self.last_actions = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float32)
        self.last_dof_vel = np.zeros((num_envs, 12), dtype=np.float32)
        self.commands = np.zeros((num_envs, 5), dtype=np.float32)
        self.ep_steps = np.zeros(num_envs, dtype=np.int32)
        self.feet_air_time = np.zeros((num_envs, 4), dtype=np.float32)
        self.last_contacts = np.zeros((num_envs, 4), dtype=np.bool_)
        self.torques = np.zeros((num_envs, 12), dtype=np.float32)
        self.foot_positions_prev = np.zeros((num_envs, 4, 3), dtype=np.float32)

        buf_len = DR['obs_latency_max'] + 1
        self.obs_buffer = np.zeros((num_envs, buf_len, NUM_OBS), dtype=np.float32)
        self.obs_buf_idx = np.zeros(num_envs, dtype=np.int32)
        self.obs_latency = np.zeros(num_envs, dtype=np.int32)

        act_buf_len = DR['action_latency_max'] + 1
        self.action_buffer = np.zeros((num_envs, act_buf_len, NUM_ACTIONS), dtype=np.float32)
        self.act_buf_idx = np.zeros(num_envs, dtype=np.int32)
        self.action_latency = np.zeros(num_envs, dtype=np.int32)

        self.env_kp = np.full(num_envs, 20.0, dtype=np.float32)
        self.env_kd = np.full(num_envs, 0.5, dtype=np.float32)
        self.motor_offset = np.zeros((num_envs, 12), dtype=np.float32)

        self.reset(np.arange(num_envs))

    def _randomize_dr(self, env_ids):
        for i in env_ids:
            self.env_kp[i] = np.random.uniform(*DR['kp_range'])
            self.env_kd[i] = np.random.uniform(*DR['kd_range'])
            self.motor_offset[i] = np.random.uniform(*DR['motor_offset_range'], size=12)
            self.obs_latency[i] = np.random.randint(0, DR['obs_latency_max'] + 1)
            self.action_latency[i] = np.random.randint(0, DR['action_latency_max'] + 1)
            self.obs_buffer[i] = 0
            self.action_buffer[i] = 0
            self.obs_buf_idx[i] = 0
            self.act_buf_idx[i] = 0

    def _get_state(self, i):
        d = self.datas[i]
        quat = d.qpos[3:7].copy()
        rot = np.zeros(9); mujoco.mju_quat2Mat(rot, quat); R = rot.reshape(3,3)
        lin_vel = R.T @ d.qvel[:3]
        ang_vel = R.T @ d.qvel[3:6]
        qw,qx,qy,qz = quat
        gravity = np.array([2*(-qz*qx+qw*qy), -2*(qz*qy+qw*qx), 1-2*(qw*qw+qz*qz)])
        return lin_vel, ang_vel, gravity, d.qpos[7:].copy(), d.qvel[6:].copy(), d.qpos[2]

    def _resample_commands(self, env_ids):
        for i in env_ids:
            self.commands[i,0] = np.random.uniform(*CMD_RANGES['vx'])
            self.commands[i,1] = np.random.uniform(*CMD_RANGES['vy'])
            self.commands[i,3] = np.random.uniform(*CMD_RANGES['heading'])
            self.commands[i,4] = np.random.uniform(*CMD_RANGES['height'])
            if np.sqrt(self.commands[i,0]**2 + self.commands[i,1]**2) < 0.2:
                self.commands[i,:2] = 0

    def reset(self, env_ids):
        for i in env_ids:
            d = self.datas[i]
            mujoco.mj_resetData(self.model, d)
            d.qpos[2] = 0.25
            d.qpos[3:7] = [1,0,0,0]
            d.qpos[7:] = DEFAULT_ANGLES * np.random.uniform(0.8, 1.2, 12)
            d.qvel[:] = 0
            mujoco.mj_forward(self.model, d)
            self.actions[i] = 0
            self.last_actions[i] = 0
            self.last_dof_vel[i] = 0
            self.ep_steps[i] = 0
            self.feet_air_time[i] = 0
            self.last_contacts[i] = True
            for fi, fid in enumerate(self.foot_body_ids):
                self.foot_positions_prev[i, fi] = d.xpos[fid]
        self._randomize_dr(env_ids)
        self._resample_commands(env_ids)
        self._compute_obs(env_ids)

    def step(self, actions_np):
        self.last_actions[:] = self.actions
        self.actions[:] = actions_np

        for i in range(self.n):
            buf_len = DR['action_latency_max'] + 1
            idx = self.act_buf_idx[i] % buf_len
            self.action_buffer[i, idx] = actions_np[i]
            self.act_buf_idx[i] += 1
            delay = self.action_latency[i]
            delayed_idx = (self.act_buf_idx[i] - 1 - delay) % buf_len
            delayed_action = self.action_buffer[i, delayed_idx]
            target = delayed_action * ACTION_SCALE + DEFAULT_ANGLES

            d = self.datas[i]
            kp, kd = self.env_kp[i], self.env_kd[i]
            for _ in range(DECIMATION):
                tau = kp * (target - d.qpos[7:]) + kd * (0 - d.qvel[6:])
                d.ctrl[:] = tau
                mujoco.mj_step(self.model, d)
            self.torques[i] = kp * (target - d.qpos[7:]) + kd * (0 - d.qvel[6:])

            if self.ep_steps[i] > 0 and self.ep_steps[i] % DR['push_interval'] == 0:
                d.qvel[:3] += np.random.uniform(*DR['push_vel_range'], size=3)

        self.ep_steps += 1
        rewards = self._compute_rewards()
        dones = self._check_termination()
        timeouts = self.ep_steps >= EPISODE_LENGTH
        dones |= timeouts

        reset_ids = np.where(dones)[0]
        if len(reset_ids) > 0:
            self.reset(reset_ids)

        self._compute_obs(np.arange(self.n))
        return self.obs.copy(), rewards, dones, {'time_outs': timeouts}

    def _compute_obs(self, env_ids):
        noise = DR['obs_noise']
        for i in env_ids:
            lv, av, g, qj, dqj, _ = self._get_state(i)

            qj_noisy = qj + self.motor_offset[i] + np.random.normal(0, noise['dof_pos'], 12)
            dqj_noisy = dqj + np.random.normal(0, noise['dof_vel'], 12)
            lv_noisy = lv + np.random.normal(0, noise['lin_vel'], 3)
            av_noisy = av + np.random.normal(0, noise['ang_vel'], 3)
            g_noisy = g + np.random.normal(0, noise['gravity'], 3)

            cmd = self.commands[i, :3].copy()
            d = self.datas[i]
            heading_err = self.commands[i,3] - np.arctan2(
                2*(d.qpos[3]*d.qpos[6]+d.qpos[4]*d.qpos[5]),
                1-2*(d.qpos[5]**2+d.qpos[6]**2))
            heading_err = (heading_err + np.pi) % (2*np.pi) - np.pi
            cmd[2] = np.clip(heading_err, -1, 1)
            self.commands[i,2] = cmd[2]

            raw_obs = np.zeros(NUM_OBS, dtype=np.float32)
            raw_obs[:3] = lv_noisy * OBS_SCALES['lin_vel']
            raw_obs[3:6] = av_noisy * OBS_SCALES['ang_vel']
            raw_obs[6:9] = g_noisy
            raw_obs[9:12] = cmd * CMD_SCALE
            raw_obs[12] = self.commands[i, 4]
            raw_obs[13:25] = (qj_noisy - DEFAULT_ANGLES) * OBS_SCALES['dof_pos']
            raw_obs[25:37] = dqj_noisy * OBS_SCALES['dof_vel']
            raw_obs[37:49] = self.actions[i]

            buf_len = DR['obs_latency_max'] + 1
            idx = self.obs_buf_idx[i] % buf_len
            self.obs_buffer[i, idx] = raw_obs
            self.obs_buf_idx[i] += 1
            delay = self.obs_latency[i]
            delayed_idx = (self.obs_buf_idx[i] - 1 - delay) % buf_len
            self.obs[i] = self.obs_buffer[i, delayed_idx]

    def _compute_rewards(self):
        rewards = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            lv, av, g, qj, dqj, h = self._get_state(i)
            cmd = self.commands[i, :3]
            r = {}
            r['tracking_lin_vel'] = np.exp(-np.sum((cmd[:2] - lv[:2])**2) / TRACKING_SIGMA)
            r['tracking_ang_vel'] = np.exp(-(cmd[2] - av[2])**2 / TRACKING_SIGMA)
            r['lin_vel_z'] = lv[2]**2
            r['ang_vel_xy'] = np.sum(av[:2]**2)
            r['orientation'] = np.sum(g[:2]**2)
            r['base_height'] = (h - self.commands[i, 4])**2
            r['torques'] = np.sum(self.torques[i]**2)
            acc = (dqj - self.last_dof_vel[i]) / POLICY_DT
            r['dof_acc'] = np.sum(acc**2)
            r['action_rate'] = np.sum((self.actions[i] - self.last_actions[i])**2)

            out = -np.clip(qj - self.dof_pos_limits[:,0], None, 0)
            out += np.clip(qj - self.dof_pos_limits[:,1], 0, None)
            r['dof_pos_limits'] = np.sum(out)

            d = self.datas[i]
            foot_z = np.array([d.xpos[fid][2] for fid in self.foot_body_ids])
            contact = foot_z < 0.025
            contact_filt = contact | self.last_contacts[i]
            first_contact = (self.feet_air_time[i] > 0) & contact_filt
            self.feet_air_time[i] += POLICY_DT
            air_rew = np.sum((self.feet_air_time[i] - 0.05) * first_contact)
            if np.linalg.norm(cmd[:2]) < 0.1: air_rew = 0
            self.feet_air_time[i] *= ~contact_filt
            self.last_contacts[i] = contact
            r['feet_air_time'] = air_rew

            foot_vel = np.zeros(4)
            for fi, fid in enumerate(self.foot_body_ids):
                vel = (d.xpos[fid] - self.foot_positions_prev[i, fi]) / POLICY_DT
                foot_vel[fi] = np.linalg.norm(vel)
                self.foot_positions_prev[i, fi] = d.xpos[fid].copy()
            r['foot_slip'] = np.sum(np.sqrt(foot_vel) * contact)

            thigh_z = np.array([d.xpos[tid][2] for tid in self.thigh_body_ids])
            calf_z = np.array([d.xpos[cid][2] for cid in self.calf_body_ids])
            r['collision'] = np.sum(thigh_z < 0.04) + np.sum(calf_z < 0.03)
            r['default_joint_pos'] = np.sum((qj - DEFAULT_ANGLES)**2)

            self.last_dof_vel[i] = dqj.copy()
            total = sum(REWARD_SCALES[k] * r[k] for k in REWARD_SCALES if k in r)
            rewards[i] = max(total, 0)
        return rewards

    def _check_termination(self):
        dones = np.zeros(self.n, dtype=np.bool_)
        for i in range(self.n):
            d = self.datas[i]
            if d.qpos[2] < 0.08:
                dones[i] = True
            _, _, g, _, _, _ = self._get_state(i)
            if abs(g[2]) < 0.3:
                dones[i] = True
        return dones

    def get_observations(self):
        return torch.from_numpy(self.obs)


def train(num_iters=500, num_steps=24, save_interval=50):
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = 'cpu'

    model = ActorCritic(NUM_OBS, NUM_OBS, NUM_ACTIONS, [512,256,128], [512,256,128], 'elu', 1.0)
    if PRETRAINED and os.path.exists(PRETRAINED):
        ckpt = torch.load(PRETRAINED, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded pretrained: {PRETRAINED}")
    model.to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    gamma, lam = 0.99, 0.95
    clip_param = 0.2
    entropy_coef = 0.01
    value_coef = 1.0
    num_epochs = 5
    num_minibatches = 4
    desired_kl = 0.01
    lr = 1e-5

    env = MuJoCoVecEnv(NUM_ENVS)
    print(f"MuJoCo VecEnv+DR: {NUM_ENVS} envs, dt={SIM_DT}, dec={DECIMATION}")
    print(f"DR: obs_latency≤{DR['obs_latency_max']}, act_latency≤{DR['action_latency_max']}, "
          f"kp={DR['kp_range']}, kd={DR['kd_range']}, offset={DR['motor_offset_range']}")

    for iteration in range(num_iters):
        t0 = time.time()
        obs_buf = torch.zeros(num_steps, NUM_ENVS, NUM_OBS)
        act_buf = torch.zeros(num_steps, NUM_ENVS, NUM_ACTIONS)
        rew_buf = torch.zeros(num_steps, NUM_ENVS)
        done_buf = torch.zeros(num_steps, NUM_ENVS)
        val_buf = torch.zeros(num_steps, NUM_ENVS)
        logp_buf = torch.zeros(num_steps, NUM_ENVS)
        mu_buf = torch.zeros(num_steps, NUM_ENVS, NUM_ACTIONS)
        sigma_buf = torch.zeros(num_steps, NUM_ENVS, NUM_ACTIONS)

        obs_t = env.get_observations().float()
        for step in range(num_steps):
            with torch.no_grad():
                mean = 4.0 * torch.tanh(model.actor(obs_t))
                std = model.std.expand_as(mean)
                dist = Normal(mean, std)
                actions = dist.sample()
                logp = dist.log_prob(actions).sum(-1)
                value = model.critic(obs_t).squeeze(-1)

            obs_buf[step] = obs_t
            act_buf[step] = actions
            val_buf[step] = value
            logp_buf[step] = logp
            mu_buf[step] = mean
            sigma_buf[step] = std

            obs_np, rewards, dones, infos = env.step(actions.numpy())
            rew_buf[step] = torch.from_numpy(rewards)
            done_buf[step] = torch.from_numpy(dones.astype(np.float32))
            obs_t = torch.from_numpy(obs_np).float()

        with torch.no_grad():
            last_val = model.critic(obs_t).squeeze(-1)

        advantages = torch.zeros(num_steps, NUM_ENVS)
        last_gae = 0
        for t in reversed(range(num_steps)):
            next_val = last_val if t == num_steps - 1 else val_buf[t+1]
            delta = rew_buf[t] + gamma * next_val * (1 - done_buf[t]) - val_buf[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - done_buf[t]) * last_gae
        returns = advantages + val_buf

        obs_flat = obs_buf.reshape(-1, NUM_OBS)
        act_flat = act_buf.reshape(-1, NUM_ACTIONS)
        logp_flat = logp_buf.reshape(-1)
        ret_flat = returns.reshape(-1)
        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        mu_flat = mu_buf.reshape(-1, NUM_ACTIONS)
        sigma_flat = sigma_buf.reshape(-1, NUM_ACTIONS)

        total = obs_flat.shape[0]
        batch_size = total // num_minibatches

        for epoch in range(num_epochs):
            perm = torch.randperm(total)
            for mb in range(num_minibatches):
                idx = perm[mb*batch_size:(mb+1)*batch_size]
                mb_obs = obs_flat[idx]
                mb_act = act_flat[idx]
                mb_logp = logp_flat[idx]
                mb_ret = ret_flat[idx]
                mb_adv = adv_flat[idx]

                mean_new = 4.0 * torch.tanh(model.actor(mb_obs))
                std_new = model.std.expand_as(mean_new)
                dist_new = Normal(mean_new, std_new)
                logp_new = dist_new.log_prob(mb_act).sum(-1)
                entropy = dist_new.entropy().sum(-1)
                val_new = model.critic(mb_obs).squeeze(-1)

                ratio = torch.exp(logp_new - mb_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-clip_param, 1+clip_param) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((val_new - mb_ret)**2).mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            with torch.no_grad():
                mean_new2 = 4.0 * torch.tanh(model.actor(obs_flat))
                std_new2 = model.std.expand_as(mean_new2)
                kl = torch.sum(torch.log(std_new2/sigma_flat+1e-5) + (sigma_flat**2+(mu_flat-mean_new2)**2)/(2*std_new2**2) - 0.5, dim=-1).mean()
                if kl > desired_kl * 2:
                    lr = max(1e-6, lr / 1.5)
                elif kl < desired_kl / 2 and kl > 0:
                    lr = min(1e-3, lr * 1.5)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

        elapsed = time.time() - t0
        mean_rew = rew_buf.sum(0).mean().item()
        mean_std = model.std.mean().item()
        print(f"Iter {iteration:4d}/{num_iters} | reward={mean_rew:6.2f} | std={mean_std:.3f} | lr={lr:.1e} | {elapsed:.2f}s")

        if (iteration + 1) % save_interval == 0:
            path = os.path.join(SAVE_DIR, f'model_{iteration+1}.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, path)
            print(f"  Saved: {path}")

    path = os.path.join(SAVE_DIR, 'model_final.pt')
    torch.save({'model_state_dict': model.state_dict()}, path)
    print(f"Final model: {path}")

    class ActorWithTanh(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor
        def forward(self, x):
            return 4.0 * torch.tanh(self.actor(x))
    wrapped = ActorWithTanh(copy.deepcopy(model.actor))
    policy_path = os.path.join(SAVE_DIR, 'policy_final.pt')
    torch.jit.save(torch.jit.script(wrapped), policy_path)
    print(f"Exported policy: {policy_path}")


if __name__ == '__main__':
    train(num_iters=500, num_steps=24, save_interval=100)
