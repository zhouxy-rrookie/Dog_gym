import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class HIMEncoder(nn.Module):
    """Hybrid Internal Model encoder.
    
    Takes history of proprioceptive observations and outputs
    velocity estimate + implicit response embedding.
    """
    def __init__(self, obs_dim, history_len=5, latent_dim=128, hidden_dims=[128, 64]):
        super().__init__()
        input_dim = obs_dim * history_len
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.vel_head = nn.Linear(hidden_dims[-1], 3)
        self.latent_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.embedding_dim = 3 + latent_dim
        
        # Projector for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
        )
    
    def forward(self, history_obs):
        feat = self.encoder(history_obs)
        vel = self.vel_head(feat)
        lat = self.latent_head(feat)
        embedding = torch.cat([vel, lat], dim=-1)
        return vel, lat, embedding
    
    def compute_contrastive_loss(self, history_batch, positive_mask, temperature=0.1):
        """InfoNCE contrastive loss.
        
        Args:
            history_batch: (B, obs_dim * history_len) 
            positive_mask: (B, B) bool - True for positive pairs (same env, adjacent timesteps)
        """
        _, _, emb = self.forward(history_batch)
        proj = F.normalize(self.projector(emb), dim=-1)
        
        logits = torch.mm(proj, proj.T) / temperature
        labels = torch.arange(history_batch.shape[0], device=history_batch.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class ActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 use_him=False,
                 him_history_len=5,
                 him_latent_dim=128,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()
        
        self.use_him = use_him
        activation = get_activation(activation)
        
        # HIM encoder
        if use_him:
            self.him = HIMEncoder(
                obs_dim=num_actor_obs,
                history_len=him_history_len,
                latent_dim=him_latent_dim,
            )
            him_embed_dim = self.him.embedding_dim
            mlp_input_dim_a = num_actor_obs + him_embed_dim
            print(f"HIM enabled: obs={num_actor_obs}, history={him_history_len}, embed={him_embed_dim}")
            print(f"Actor input dim: {mlp_input_dim_a}")
        else:
            self.him = None
            mlp_input_dim_a = num_actor_obs
        
        mlp_input_dim_c = num_critic_obs
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
    
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
    
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations, history_obs=None):
        if self.use_him:
            if history_obs is not None:
                _, _, him_embed = self.him(history_obs)
            else:
                him_embed = torch.zeros(observations.shape[0], self.him.embedding_dim, device=observations.device)
            actor_input = torch.cat([observations, him_embed], dim=-1)
        else:
            actor_input = observations
        mean = 4.0 * torch.tanh(self.actor(actor_input))
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, observations, history_obs=None, **kwargs):
        self.update_distribution(observations, history_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations, history_obs=None, **kwargs):
        if self.use_him and history_obs is not None:
            _, _, him_embed = self.him(history_obs)
            actor_input = torch.cat([observations, him_embed], dim=-1)
        else:
            actor_input = observations
        actions_mean = 4.0 * torch.tanh(self.actor(actor_input))
        return actions_mean
    
    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
    
    def compute_him_contrastive_loss(self, history_obs_batch):
        """HIO contrastive loss for HIM encoder."""
        if not self.use_him:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.him.compute_contrastive_loss(history_obs_batch, None)
