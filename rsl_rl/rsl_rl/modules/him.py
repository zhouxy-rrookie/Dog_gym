import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridInternalModel(nn.Module):
    """HIM encoder from 'Hybrid Internal Model' paper.
    
    Takes history of proprioceptive observations and outputs:
    - velocity estimate v_hat (3-dim)
    - implicit latent l_hat (128-dim)
    Together these form the hybrid internal embedding.
    """
    def __init__(self, obs_dim=45, history_len=5, latent_dim=128, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.latent_dim = latent_dim
        input_dim = obs_dim * history_len
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        self.encoder = nn.Sequential(*layers)
        
        # Output heads
        self.vel_head = nn.Linear(hidden_dims[-1], 3)      # explicit velocity estimate
        self.latent_head = nn.Linear(hidden_dims[-1], latent_dim)  # implicit response
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(3 + latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
        )
        
        # Target encoder (EMA updated)
        self.target_encoder = nn.Sequential(*layers)
        self.target_vel_head = nn.Linear(hidden_dims[-1], 3)
        self.target_latent_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.target_projection = nn.Sequential(
            nn.Linear(3 + latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
        )
        
        # Initialize target encoder with same weights
        self._copy_weights()
        
        # Disable gradients for target
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_vel_head.parameters():
            p.requires_grad = False
        for p in self.target_latent_head.parameters():
            p.requires_grad = False
        for p in self.target_projection.parameters():
            p.requires_grad = False
    
    def _copy_weights(self):
        for src, dst in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            dst.data.copy_(src.data)
        for src, dst in zip(self.vel_head.parameters(), self.target_vel_head.parameters()):
            dst.data.copy_(src.data)
        for src, dst in zip(self.latent_head.parameters(), self.target_latent_head.parameters()):
            dst.data.copy_(src.data)
        for src, dst in zip(self.projection.parameters(), self.target_projection.parameters()):
            dst.data.copy_(src.data)
    
    @torch.no_grad()
    def update_target(self, tau=0.99):
        for src, dst in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            dst.data = tau * dst.data + (1 - tau) * src.data
        for src, dst in zip(self.vel_head.parameters(), self.target_vel_head.parameters()):
            dst.data = tau * dst.data + (1 - tau) * src.data
        for src, dst in zip(self.latent_head.parameters(), self.target_latent_head.parameters()):
            dst.data = tau * dst.data + (1 - tau) * src.data
        for src, dst in zip(self.projection.parameters(), self.target_projection.parameters()):
            dst.data = tau * dst.data + (1 - tau) * src.data
    
    def get_embedding(self, history_obs):
        """Get hybrid internal embedding from history.
        
        Args:
            history_obs: (B, obs_dim * history_len) tensor of concatenated history observations
        
        Returns:
            velocity_hat: (B, 3) estimated velocity
            latent_hat: (B, latent_dim) implicit response
            embedding: (B, 3 + latent_dim) concatenated
        """
        feat = self.encoder(history_obs)
        velocity_hat = self.vel_head(feat)
        latent_hat = self.latent_head(feat)
        embedding = torch.cat([velocity_hat, latent_hat], dim=-1)
        return velocity_hat, latent_hat, embedding
    
    def project(self, embedding):
        return F.normalize(self.projection(embedding), dim=-1)
    
    @torch.no_grad()
    def project_target(self, embedding):
        return F.normalize(self.target_projection(embedding), dim=-1)
    
    def contrastive_loss(self, history_obs, future_history_obs, temperature=0.1):
        """Compute contrastive loss.
        
        Pulls the embedding from current history close to the embedding from future history
        (of the same env), and pushes away from other envs' embeddings.
        
        Args:
            history_obs: (B, obs_dim * history_len) current history
            future_history_obs: (B, obs_dim * history_len) future history (successor state)
        """
        B = history_obs.shape[0]
        
        _, _, embedding = self.get_embedding(history_obs)
        z = self.project(embedding)  # (B, 128)
        
        with torch.no_grad():
            _, _, future_embedding = self.get_embedding(future_history_obs)
            # Use target encoder for stability
            feat_t = self.target_encoder(future_history_obs)
            vt = self.target_vel_head(feat_t)
            lt = self.target_latent_head(feat_t)
            future_embedding_t = torch.cat([vt, lt], dim=-1)
            z_t = self.project_target(future_embedding_t)  # (B, 128)
        
        # Cosine similarity matrix
        logits = torch.mm(z, z_t.T) / temperature  # (B, B)
        labels = torch.arange(B, device=history_obs.device)
        
        # Symmetric loss
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss
