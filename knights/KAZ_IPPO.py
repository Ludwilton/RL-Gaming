import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from datetime import datetime
from pettingzoo.butterfly import knights_archers_zombies_v10
from collections import deque
import time
import pygame

"""
# vibes
"""

class EntityEncoder(nn.Module):
    def __init__(self, input_dim, enc_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_dim),
            nn.Tanh(),
            nn.Linear(enc_dim, enc_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)

class SetNet(nn.Module):
    def __init__(self, obs_structure, enc_dim, hidden_dim, out_dim, mask_obs):
        super().__init__()
        self.obs_structure = obs_structure
        self.entity_encoders = nn.ModuleList([
            EntityEncoder(obs_dim, enc_dim) for _, obs_dim in obs_structure
        ])
        self.head = nn.Sequential(
            nn.Linear(enc_dim * len(obs_structure), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.mask_obs = mask_obs

    def generate_mask(self, x, segment_length):
        batch_size = x.shape[0]
        num_segments = x.shape[1] // segment_length
        x_reshaped = x.view(batch_size, num_segments, segment_length)
        mask = torch.any(x_reshaped != 0, dim=2)
        return mask.float()

    def forward(self, x):
        batch_size = x.shape[0]
        pooled_entity_features = []
        current_position = 0
        
        for encoder_idx, (num_entities, obs_dim) in enumerate(self.obs_structure):
            segment_length = num_entities * obs_dim
            entity_observations = x[:, current_position:current_position + segment_length].reshape(-1, obs_dim)
            
            entity_encoded = self.entity_encoders[encoder_idx](entity_observations).reshape(
                batch_size, num_entities, -1
            )

            if self.mask_obs:
                mask = self.generate_mask(
                    x[:, current_position:current_position + segment_length], obs_dim
                ).reshape(batch_size, num_entities, 1)
                entity_encoded = entity_encoded * mask
            
            # Pool: mean + max
            entity_features = torch.mean(entity_encoded, dim=1) + torch.max(entity_encoded, dim=1)[0]
            pooled_entity_features.append(entity_features)
            current_position += segment_length
        
        concatenated_features = torch.cat(pooled_entity_features, dim=1)
        return self.head(concatenated_features)

class SetActorCritic(nn.Module):
    def __init__(self, obs_structure, enc_dim, hidden_dim, action_dim, mask_obs):
        super().__init__()
        self.critic = SetNet(obs_structure, enc_dim, hidden_dim, 1, mask_obs)
        self.actor = SetNet(obs_structure, enc_dim, hidden_dim, action_dim, mask_obs)
        self.actor.head[-1].weight.data *= 0.01
    
    def forward(self, x):
        return self.actor(x), self.critic(x)


class IPPO:
    def __init__(self, config):
        self.env = config['env']
        self.device = config['device']
        self.config = config
        self.initialize_agents()
        
    def set_network(self, agent):
        if self.config['network_groups']:
            for group in self.config['network_groups']:
                if agent in group:
                    for other_agent in group:
                        if other_agent in self.agents:
                            return self.agents[other_agent]['network'], self.agents[other_agent]['optimizer']
                    break
        
        state_dim = np.prod(self.env.observation_space(agent).shape)
        action_dim = self.env.action_space(agent).n
        
        if self.config['setnet']:
            network = SetActorCritic(
                self.config['obs_structure'], 
                self.config['enc_dim'],
                self.config['hidden_dim'], 
                action_dim, 
                self.config['mask_obs']
            ).to(self.device)
        else:
            network = nn.Sequential(
                nn.Linear(state_dim, self.config['hidden_dim']),
                nn.Tanh(),
                nn.Linear(self.config['hidden_dim'], action_dim)
            ).to(self.device)
        
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config['lr'])
        return network, optimizer

    def initialize_agents(self):
        self.agents = {}
        for agent in self.env.possible_agents:
            network, optimizer = self.set_network(agent)
            buffer = deque(maxlen=self.config['rollout_steps'])
            self.agents[agent] = {
                'network': network, 
                'optimizer': optimizer, 
                'buffer': buffer
            }
        os.makedirs(self.config['save_dir'], exist_ok=True)
    
    def select_action(self, agent, state, deterministic=False):
        network = self.agents[agent]['network']
        with torch.no_grad():
            state_tensor = torch.tensor(
                state.reshape(-1), 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)
            
            logits = network.actor(state_tensor)
            
            if deterministic:
                return logits.argmax(1).item()
            
            return torch.distributions.Categorical(logits=logits).sample().item()
    
    def step(self, agent, state, action, reward, next_state, termination, truncation):
        buffer = self.agents[agent]['buffer']
        
        state = state.reshape(-1)
        next_state = next_state.reshape(-1)
        buffer.append((state, action, reward, next_state, termination, truncation))
        
        if len(buffer) == self.config['rollout_steps']:
            self.learn(agent)
            buffer.clear()
    
    def learn(self, agent):
        buffer = self.agents[agent]['buffer']
        network = self.agents[agent]['network']
        optimizer = self.agents[agent]['optimizer']
        
        states, actions, rewards, next_states, terminations, truncations = zip(*buffer)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        terminations = torch.tensor(np.array(terminations), dtype=torch.float32, device=self.device)
        truncations = torch.tensor(np.array(truncations), dtype=torch.float32, device=self.device)
        
        if self.config['clip_rewards']:
            rewards = torch.clamp(rewards, -self.config['reward_clip'], self.config['reward_clip'])
        if self.config['scale_rewards']:
            rewards = rewards / (rewards.std() + 1e-8)
        
        with torch.no_grad():
            logits = network.actor(states)
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
            values = network.critic(states).squeeze(1)
            
            next_values = torch.cat((values[1:], network.critic(next_states[-1:]).squeeze(0)), dim=0)
            advantages = torch.zeros_like(rewards, device=self.device)
            advantage = 0.
            
            for t in reversed(range(self.config['rollout_steps'])):
                non_termination = 1. - terminations[t]
                non_truncation = 1. - truncations[t]
                delta = rewards[t] + self.config['gamma'] * next_values[t] * non_termination - values[t]
                advantages[t] = advantage = delta + self.config['gamma'] * self.config['lambda'] * \
                    non_termination * non_truncation * advantage
            
            returns = advantages + values
            
            if self.config['advantage_norm']:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        b_indices = np.arange(len(states))
        minibatch_size = len(states) // self.config['num_minibatches']
        
        for epoch in range(self.config['num_epochs']):
            np.random.shuffle(b_indices)
            
            for minibatch in range(self.config['num_minibatches']):
                start = minibatch * minibatch_size
                end = start + minibatch_size
                mb_indices = b_indices[start:end]
                
                # Forward pass
                new_logits, new_values = network(states[mb_indices])
                
                # Policy loss
                distribution = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = distribution.log_prob(actions[mb_indices])
                ratio = (new_log_probs - log_probs[mb_indices]).exp()
                
                surr1 = ratio * advantages[mb_indices]
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config['ppo_clip'], 
                    1 + self.config['ppo_clip']
                ) * advantages[mb_indices]
                loss_policy = -torch.min(surr1, surr2).mean()
                
                # Value loss
                loss_value = F.mse_loss(new_values.squeeze(1), returns[mb_indices])
                
                # Entropy
                entropy = distribution.entropy().mean()
                
                # Combined loss
                loss = loss_policy + self.config['value_loss'] * loss_value - \
                       self.config['entropy_beta'] * entropy
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), self.config['grad_norm_clip'])
                optimizer.step()
    
    def train(self):
        """Main training loop"""
        if self.config['verbose']:
            print("Training IPPO with SetNet\n")
        
        logs = {
            'episode_count': 0,
            'episodic_reward': 0.,
            'episode_rewards': [],
            'steps': 0,
            'start_time': time.time()
        }
        
        while logs['steps'] < self.config['total_steps']:
            # Reset environment
            observations, infos = self.env.reset()
            
            while self.env.agents:
                active_agents = self.env.agents
                
                # Get actions
                actions = {
                    agent: self.select_action(agent, observations[agent]) 
                    for agent in active_agents
                }
                
                # Environment step
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                # Agent steps
                for agent in active_agents:
                    self.step(
                        agent,
                        observations[agent],
                        actions[agent],
                        rewards[agent],
                        next_observations[agent],
                        terminations[agent],
                        truncations[agent]
                    )
                
                observations = next_observations
                logs['steps'] += 1
                logs['episodic_reward'] += sum(rewards.values())
            
            logs['episode_count'] += 1
            logs['episode_rewards'].append(logs['episodic_reward'])
            logs['episodic_reward'] = 0.
            
            # Progress printing
            if self.config['verbose']:
                avg_reward = np.mean(logs['episode_rewards'][-20:])
                print(f"\r--- {100 * logs['steps'] / self.config['total_steps']:.1f}%"
                      f"\t Step: {logs['steps']:,}"
                      f"\t Mean Reward: {avg_reward:.2f}"
                      f"\t Episode: {logs['episode_count']:,}"
                      f"\t Duration: {time.time() - logs['start_time']:,.1f}s ---", end='')
                
                # Save checkpoints
                if logs['episode_count'] % 50 == 0:
                    for agent in self.agents:
                        model_path = os.path.join(
                            self.config['save_dir'],
                            f'model_{agent}_episode_{logs["episode_count"]}.pth'
                        )
                        torch.save(self.agents[agent]['network'].state_dict(), model_path)
                    print()
            
            # Check target reward
            if self.config['target_reward'] and \
               np.mean(logs['episode_rewards'][-20:]) >= self.config['target_reward']:
                break
        
        if self.config['verbose']:
            print("\n\nTraining complete!")
        
        logs['end_time'] = time.time()
        logs['duration'] = logs['end_time'] - logs['start_time']
        return logs

def get_config():
    return {
        'env': knights_archers_zombies_v10.parallel_env(
            spawn_rate=20,
            num_archers=2,
            num_knights=2,
            max_zombies=20,
            max_arrows=30,
            killable_knights=True,
            killable_archers=True,
            line_death=True,
            max_cycles=1200,
            vector_state=True
        ),
        'network_groups': (('archer_0', 'archer_1'), ('knight_0', 'knight_1')),
        'obs_structure': ((1, 5), (2, 5), (2, 5), (2, 5), (20, 5), (20, 5)),
        'setnet': True,
        'mask_obs': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'total_steps': 5000000,
        'target_reward': 100,
        'entropy_beta': 0.01,
        'gamma': 0.99,
        'lambda': 0.95,
        'lr': 1e-4,
        'hidden_dim': 512,
        'rollout_steps': 2048,
        'num_epochs': 10,
        'num_minibatches': 1,
        'advantage_norm': True,
        'value_loss': 1,
        'ppo_clip': 0.15,
        'grad_norm_clip': 0.5,
        'clip_rewards': False,
        'reward_clip': 20,
        'scale_rewards': False,
        'verbose': True,
        'save_dir': 'models_ippo'
    }



def evaluate(ippo, num_games=5, render=True):
    env = knights_archers_zombies_v10.parallel_env(
        render_mode='human' if render else None,
        spawn_rate=20,
        num_archers=2,
        num_knights=2,
        max_zombies=20,
        max_arrows=30,
        killable_knights=True,
        killable_archers=True,
        line_death=True,
        max_cycles=3600,
        vector_state=True,
        use_typemasks=False
    )
    
    for game in range(num_games):
        observations, infos = env.reset(seed=1000 + game)
        total_reward = 0
        step_count = 0
        
        print(f"\n--- Game {game + 1}/{num_games} ---")
        
        while env.agents:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nEvaluation interrupted by user")
                        env.close()
                        return
            
            actions = {
                agent: ippo.select_action(agent, observations[agent], deterministic=True)
                for agent in env.agents
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())
            step_count += 1
        
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps Survived: {step_count}")
    
    env.close()
    print("\nEvaluation complete!")

def main():
    config = get_config()
    
    print(f"\n{'='*60}")
    print(f"  IPPO with SetNet - {config['device'].upper()}")
    print(f"{'='*60}\n")
    
    mode = input("Select mode - (1) Train, (2) Evaluate: ").strip()
    
    if mode == "1":
        ippo = IPPO(config)
        logs = ippo.train()
        print(f"\nTraining completed in {logs['duration']:.1f}s")
        
    elif mode == "2":
        ippo = IPPO(config)
        
        episode = int(input("Enter episode number to load: "))
        loaded_count = 0
        for agent in ippo.agents:
            model_path = os.path.join(config['save_dir'], f'model_{agent}_episode_{episode}.pth')
            if os.path.exists(model_path):
                ippo.agents[agent]['network'].load_state_dict(
                    torch.load(model_path, map_location=config['device'], weights_only=True)
                )
                print(f"Loaded {agent} from episode {episode}")
                loaded_count += 1
            else:
                print(f"WARNING: No model found for {agent} at episode {episode}")
        
        if loaded_count == 0:
            print("\nERROR: No models were loaded! Check the episode number and save directory.")
            return
        
        print(f"\nSuccessfully loaded {loaded_count}/{len(ippo.agents)} agents")
        print("\nStarting evaluation... (Close the pygame window to stop)")
        evaluate(ippo, num_games=5, render=True)
    
    else:
        print("Invalid mode selected. Please choose 1 or 2.")


if __name__ == "__main__":
    main()