"""Deploy 46-dim frame-stacked policy in MuJoCo."""
import os
os.environ['MUJOCO_GL'] = 'egl'
import sys
sys.path.insert(0, '/root/gpufree-data/workspace/legged_gym')
sys.path.insert(0, '/root/gpufree-data/workspace/rsl_rl')

import copy
import mujoco
import numpy as np
import torch
from rsl_rl.modules import ActorCritic

CKPT = '/root/gpufree-data/workspace/legged_gym/logs/rough_dog_urdf/Jun20_20-19-10_/model_8350.pt'
DOG_MJCF = '/root/gpufree-data/workspace/legged_gym/resources/robots/dog_urdf/urdf/dog_with_meshes.mjcf'

SIM_DT = 0.002
CONTROL_DEC = 10
KP, KD = 20.0, 0.5
ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE, DOF_VEL_SCALE = 1.0, 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_SINGLE_OBS = 46
FRAME_STACK = 3
NUM_OBS = NUM_SINGLE_OBS * FRAME_STACK
NUM_CRITIC_OBS = 53

DEFAULT_ANGLES = np.array([
    0.1, 0.8, -1.5, 0.1, 0.8, -1.5, 0.1, 1.0, -1.5, 0.1, 1.0, -1.5,
], dtype=np.float32)


def get_gravity(quat):
    qw, qx, qy, qz = quat
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ])


def quat_rotate_inverse(quat, vel):
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    return rot.reshape(3, 3).T @ vel


def build_single_obs(d, action_prev, cmd, height_target):
    quat = d.qpos[3:7]
    ang_vel_body = quat_rotate_inverse(quat, d.qvel[3:6])
    gravity = get_gravity(quat)
    qj = d.qpos[7:]
    dqj = d.qvel[6:]

    obs = np.zeros(NUM_SINGLE_OBS, dtype=np.float32)
    obs[:3] = ang_vel_body * ANG_VEL_SCALE
    obs[3:6] = gravity
    obs[6:9] = cmd * CMD_SCALE
    obs[9] = height_target
    obs[10:22] = (qj - DEFAULT_ANGLES) * DOF_POS_SCALE
    obs[22:34] = dqj * DOF_VEL_SCALE
    obs[34:46] = action_prev
    return obs


def export_policy(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model = ActorCritic(
        num_actor_obs=NUM_OBS, num_critic_obs=NUM_CRITIC_OBS, num_actions=12,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation='elu', init_noise_std=1.0,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class ActorWithTanh(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor
        def forward(self, x):
            return 4.0 * torch.tanh(self.actor(x))

    wrapped = ActorWithTanh(copy.deepcopy(model.actor))
    out_path = ckpt_path.replace('.pt', '_policy.pt')
    torch.jit.save(torch.jit.trace(wrapped, torch.zeros(1, NUM_OBS)), out_path)
    print(f"Exported: {out_path}")
    return out_path


if __name__ == '__main__':
    print("=== Deploying 46-dim frame-stacked policy ===")

    if CKPT.endswith('_policy.pt'):
        policy = torch.jit.load(CKPT, map_location='cpu')
    else:
        policy_path = export_policy(CKPT)
        policy = torch.jit.load(policy_path, map_location='cpu')
    policy.eval()

    m = mujoco.MjModel.from_xml_path(DOG_MJCF)
    d = mujoco.MjData(m)
    m.opt.timestep = SIM_DT
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, mass={sum(m.body_mass):.1f}kg")

    d.qpos[2] = 0.25
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = DEFAULT_ANGLES
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(500):
        tau = KP * (DEFAULT_ANGLES - d.qpos[7:]) + KD * (0 - d.qvel[6:])
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    print(f"After stabilization: z={d.qpos[2]:.4f}")

    try:
        renderer = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 0
        cam.distance = 1.5
        cam.azimuth = 135
        cam.elevation = -20
    except:
        renderer, cam = None, None

    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    height_target = 0.25
    action_prev = np.zeros(12, dtype=np.float32)
    obs_history = np.zeros(NUM_OBS, dtype=np.float32)
    target = DEFAULT_ANGLES.copy()
    frames, positions = [], []
    counter = 0
    duration = 10.0

    for i in range(int(duration / SIM_DT)):
        counter += 1
        if counter % CONTROL_DEC == 0:
            single = build_single_obs(d, action_prev, cmd, height_target)
            obs_history[:NUM_SINGLE_OBS*(FRAME_STACK-1)] = obs_history[NUM_SINGLE_OBS:]
            obs_history[NUM_SINGLE_OBS*(FRAME_STACK-1):] = single

            obs_tensor = torch.from_numpy(obs_history).unsqueeze(0).float()
            with torch.no_grad():
                act = policy(obs_tensor).squeeze(0).numpy()
            target = act * ACTION_SCALE + DEFAULT_ANGLES
            action_prev[:] = act

        tau = KP * (target - d.qpos[7:]) + KD * (0 - d.qvel[6:])
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if i % int(1.0 / (SIM_DT * 30)) == 0:
            positions.append(d.qpos[:3].copy())
            if renderer:
                renderer.update_scene(d, cam)
                frames.append(renderer.render().copy())

    positions = np.array(positions)
    dist = np.sqrt(positions[-1, 0]**2 + positions[-1, 1]**2)
    print(f"\n=== Result ===")
    print(f"Final pos: ({positions[-1,0]:.2f}, {positions[-1,1]:.2f}, {positions[-1,2]:.2f})")
    print(f"Distance: {dist:.2f} m")
    print(f"Avg height: {positions[:,2].mean():.3f} m")
    print(f"Min height: {positions[:,2].min():.3f} m")
    print(f"Fell: {'YES' if positions[:,2].min() < 0.10 else 'NO'}")

    if frames:
        try:
            import cv2
            out_path = '/tmp/dog_urdf_trained_mujoco.mp4'
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"Video: {out_path}")
        except ImportError:
            pass
