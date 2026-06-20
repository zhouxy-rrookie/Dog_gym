"""Export trained policy to TorchScript."""
import torch
import copy
import os
import sys

sys.path.insert(0, '/root/gpufree-data/workspace/legged_gym')
sys.path.insert(0, '/root/gpufree-data/workspace/rsl_rl')

from rsl_rl.modules import ActorCritic

ckpt_path = '/root/gpufree-data/workspace/legged_gym/logs/rough_dog_urdf/Jun18_22-51-12_/model_4400.pt'
out_dir = '/root/gpufree-data/workspace/legged_gym/logs/rough_dog_urdf/exported/model_4400'

checkpoint = torch.load(ckpt_path, map_location='cpu')

num_actor_obs = 48
num_critic_obs = 48
num_actions = 12
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]

model = ActorCritic(
    num_actor_obs=num_actor_obs,
    num_critic_obs=num_critic_obs,
    num_actions=num_actions,
    actor_hidden_dims=actor_hidden_dims,
    critic_hidden_dims=critic_hidden_dims,
    activation='elu',
    init_noise_std=0.5,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'policy_{int(ckpt_path.split("_")[-1].split(".")[0])}.pt')

actor = copy.deepcopy(model.actor).to('cpu')
traced = torch.jit.script(actor)
torch.jit.save(traced, out_path)
print(f'Exported: {out_path}')
print(f'Input size: {num_actor_obs}, Output size: {num_actions}')
