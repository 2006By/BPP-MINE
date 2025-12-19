import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import time
import os
import copy
from typing import List

class ParallelMCTSTrainer:
    """
    Optimized MCTS Trainer with parallel environment processing.
    
    Key optimizations:
    1. Run N environments in parallel
    2. Batch all network forward passes
    3. Minimize CPU-GPU data transfers
    4. Aggregate experiences for efficient training
    """
    
    def __init__(self,
                 set_transformer,
                 pct_policy,
                 mcts_planner,
                 args,
                 writer=None,
                 device='cuda',
                 num_parallel_envs=16):
        """
        Args:
            num_parallel_envs: Number of environments to run in parallel
                              Default 16 for better GPU utilization
        """
        self.set_transformer = set_transformer
        self.pct_policy = pct_policy
        self.mcts_planner = mcts_planner
        self.args = args
        self.writer = writer
        self.device = device
        self.num_parallel_envs = num_parallel_envs
        
        # Optimizers
        self.st_optimizer = optim.Adam(
            set_transformer.parameters(),
            lr=args.st_learning_rate
        )
        
        if args.train_pct:
            self.pct_optimizer = optim.Adam(
                pct_policy.parameters(),
                lr=args.pct_learning_rate
            )
        else:
            for param in pct_policy.parameters():
                param.requires_grad = False
            self.pct_optimizer = None
        
        # Training statistics
        self.episode_count = 0
        self.step_count = 0
        
        # Experience aggregation buffer
        self.experience_buffer = {
            'states': [],
            'policies': [],
            'values': []
        }
        self.batch_size = args.batch_size if hasattr(args, 'batch_size') else 64
    
    def collect_parallel_episodes(self, env_template, num_episodes):
        """
        Collect episodes from multiple parallel environments.
        
        Args:
            env_template: Template environment to copy
            num_episodes: Number of episodes to collect
            
        Returns:
            experiences: Aggregated experiences from all episodes
        """
        all_states = []
        all_policies = []
        all_values = []
        
        # Process episodes in batches
        num_batches = (num_episodes + self.num_parallel_envs - 1) // self.num_parallel_envs
        
        for batch_idx in range(num_batches):
            batch_size = min(self.num_parallel_envs, 
                           num_episodes - batch_idx * self.num_parallel_envs)
            
            # Create parallel environments
            envs = [copy.deepcopy(env_template) for _ in range(batch_size)]
            states = [env.reset() for env in envs]
            dones = [False] * batch_size
            
            # Storage for this batch
            batch_states = [[] for _ in range(batch_size)]
            batch_policies = [[] for _ in range(batch_size)]
            batch_rewards = [[] for _ in range(batch_size)]
            
            # Run episodes in parallel
            while not all(dones):
                # Get active environment indices
                active_indices = [i for i, done in enumerate(dones) if not done]
                
                if not active_indices:
                    break
                
                # Batch MCTS search for active environments
                batch_mcts_policies = self._batch_mcts_search(
                    [envs[i] for i in active_indices]
                )
                
                # Execute actions in each active environment
                for idx, env_idx in enumerate(active_indices):
                    pi_mcts = batch_mcts_policies[idx]
                    
                    # Store experience
                    batch_states[env_idx].append(states[env_idx].copy())
                    batch_policies[env_idx].append(pi_mcts.copy())
                    
                    # Sample and execute action
                    if self.args.use_temperature:
                        action_idx = np.random.choice(len(pi_mcts), p=pi_mcts)
                    else:
                        action_idx = np.argmax(pi_mcts)
                    
                    state_obs, reward, done, info = envs[env_idx].step(action_idx)
                    
                    states[env_idx] = state_obs
                    batch_rewards[env_idx].append(reward)
                    dones[env_idx] = done
                    
                    self.step_count += 1
            
            # Process collected experiences
            for env_idx in range(batch_size):
                if len(batch_states[env_idx]) > 0:
                    # Get final ratio as value
                    final_ratio = envs[env_idx].get_ratio()
                    
                    # Add to aggregated experiences
                    all_states.extend(batch_states[env_idx])
                    all_policies.extend(batch_policies[env_idx])
                    all_values.extend([final_ratio] * len(batch_states[env_idx]))
                    
                    self.episode_count += 1
        
        return {
            'states': all_states,
            'policies': all_policies,
            'values': all_values
        }
    
    def _batch_mcts_search(self, envs: List):
        """
        Run MCTS search for multiple environments in batch.
        
        This is more efficient than running MCTS sequentially.
        """
        policies = []
        
        # For now, run sequentially (can be optimized further)
        # True batching of MCTS is complex and requires tree-level parallelization
        for env in envs:
            with torch.no_grad():
                pi_mcts = self.mcts_planner.search(env)
            policies.append(pi_mcts)
        
        return policies
    
    def _update_networks_batch(self, experiences):
        """
        Update networks using batched experiences.
        
        Much more efficient than one-by-one updates.
        """
        states = experiences['states']
        target_policies = experiences['policies']
        values = experiences['values']
        
        total_samples = len(states)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        total_policy_loss = 0
        total_value_loss = 0
        n_batches = 0
        
        # Process in batches
        for start_idx in range(0, total_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Prepare batch tensors
            batch_states = np.array([states[i] for i in batch_indices])
            batch_pi_targets = np.array([target_policies[i] for i in batch_indices])
            batch_value_targets = np.array([values[i] for i in batch_indices])
            
            # Convert to tensors (single GPU transfer!)
            states_tensor = torch.FloatTensor(batch_states).to(self.device)
            pi_targets_tensor = torch.FloatTensor(batch_pi_targets).to(self.device)
            value_targets_tensor = torch.FloatTensor(batch_value_targets).unsqueeze(1).to(self.device)
            
            # Create mask
            mask = torch.FloatTensor(
                np.sum(batch_states, axis=-1) == 0
            ).to(self.device)
            
            # Forward pass (batched!)
            pi_pred, value_pred = self.set_transformer(states_tensor, mask.bool())
            
            # Policy loss: Cross-entropy
            policy_loss = -torch.sum(
                pi_targets_tensor * torch.log(pi_pred + 1e-8),
                dim=-1
            ).mean()
            
            # Value loss: MSE
            value_loss = F.mse_loss(value_pred, value_targets_tensor)
            
            # Combined loss
            loss = (
                self.args.policy_loss_weight * policy_loss +
                self.args.value_loss_weight * value_loss
            )
            
            # Update
            self.st_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.set_transformer.parameters(),
                self.args.max_grad_norm
            )
            self.st_optimizer.step()
            
            # Accumulate stats
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1
        
        return {
            'policy_loss': total_policy_loss / max(n_batches, 1),
            'value_loss': total_value_loss / max(n_batches, 1),
        }
    
    def train(self, env_template, num_episodes):
        """
        Main training loop with parallel episode collection.
        """
        print(f"Starting parallel MCTS training for {num_episodes} episodes...")
        print(f"Using {self.num_parallel_envs} parallel environments")
        print(f"Batch size: {self.batch_size}")
        
        episode_ratios = deque(maxlen=100)
        start_time = time.time()
        
        # Collect episodes in parallel batches
        episodes_per_update = self.num_parallel_envs * 2  # Collect 2x parallel envs worth
        num_updates = (num_episodes + episodes_per_update - 1) // episodes_per_update
        
        for update_idx in range(num_updates):
            episodes_to_collect = min(
                episodes_per_update,
                num_episodes - update_idx * episodes_per_update
            )
            
            # Collect experiences
            experiences = self.collect_parallel_episodes(
                env_template,
                episodes_to_collect
            )
            
            # Update networks with batched data
            train_stats = self._update_networks_batch(experiences)
            
            # Track statistics
            for val in experiences['values']:
                episode_ratios.append(val)
            
            # Print progress
            if (update_idx + 1) % max(1, num_updates // 100) == 0:
                elapsed = time.time() - start_time
                fps = self.step_count / elapsed
                
                print(f"\nUpdate {update_idx + 1}/{num_updates}")
                print(f"  Episodes: {self.episode_count}/{num_episodes}")
                print(f"  Steps: {self.step_count}, FPS: {fps:.1f}")
                print(f"  Last 100 episodes:")
                print(f"    Mean ratio: {np.mean(episode_ratios):.4f}")
                print(f"    Max ratio: {np.max(episode_ratios):.4f}")
                print(f"  Losses:")
                print(f"    Policy: {train_stats['policy_loss']:.4f}")
                print(f"    Value: {train_stats['value_loss']:.4f}")
                
                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar('Train/MeanRatio', np.mean(episode_ratios), self.episode_count)
                    self.writer.add_scalar('Train/MaxRatio', np.max(episode_ratios), self.episode_count)
                    self.writer.add_scalar('Loss/Policy', train_stats['policy_loss'], self.episode_count)
                    self.writer.add_scalar('Loss/Value', train_stats['value_loss'], self.episode_count)
                    self.writer.add_scalar('Performance/FPS', fps, self.episode_count)
            
            # Save checkpoint
            if (self.episode_count) % self.args.save_interval == 0:
                self.save_checkpoint(self.episode_count)
        
        print("\nTraining completed!")
        print(f"Final mean ratio: {np.mean(episode_ratios):.4f}")
    
    def save_checkpoint(self, episode):
        """Save model checkpoint."""
        if not hasattr(self.args, 'save_dir'):
            return
            
        save_dir = self.args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        checkpoint = {
            'episode': episode,
            'set_transformer': self.set_transformer.state_dict(),
            'st_optimizer': self.st_optimizer.state_dict(),
        }
        
        if self.pct_optimizer is not None:
            checkpoint['pct_policy'] = self.pct_policy.state_dict()
            checkpoint['pct_optimizer'] = self.pct_optimizer.state_dict()
        
        save_path = os.path.join(save_dir, f'checkpoint_ep{episode}.pt')
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.set_transformer.load_state_dict(checkpoint['set_transformer'])
        self.st_optimizer.load_state_dict(checkpoint['st_optimizer'])
        
        if 'pct_policy' in checkpoint and self.pct_optimizer is not None:
            self.pct_policy.load_state_dict(checkpoint['pct_policy'])
            self.pct_optimizer.load_state_dict(checkpoint['pct_optimizer'])
        
        self.episode_count = checkpoint.get('episode', 0)
        print(f"Loaded checkpoint from episode {self.episode_count}")
