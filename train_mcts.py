import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import time
import os

class MCTSTrainer:
    """
    AlphaZero-style trainer for hierarchical bin packing.
    
    Combines:
    1. Set Transformer for prior estimation
    2. MCTS for lookahead planning
    3. PCT for item placement
    4. Policy distillation for learning
    """
    
    def __init__(self,
                 set_transformer,
                 pct_policy,
                 mcts_planner,
                 args,
                 writer=None,
                 device='cuda'):
        """
        Args:
            set_transformer: Set Transformer network (trainable)
            pct_policy: PCT network (can be frozen or trainable)
            mcts_planner: MCTS planner
            args: Training arguments
            writer: TensorBoard writer
            device: Device to run on
        """
        self.set_transformer = set_transformer
        self.pct_policy = pct_policy
        self.mcts_planner = mcts_planner
        self.args = args
        self.writer = writer
        self.device = device
        
        # Separate optimizers for Set Transformer and PCT
        self.st_optimizer = optim.Adam(
            set_transformer.parameters(),
            lr=args.st_learning_rate
        )
        
        if args.train_pct:
            # Train PCT alongside Set Transformer
            self.pct_optimizer = optim.Adam(
                pct_policy.parameters(),
                lr=args.pct_learning_rate
            )
        else:
            # Freeze PCT
            for param in pct_policy.parameters():
                param.requires_grad = False
            self.pct_optimizer = None
        
        # Training statistics
        self.episode_count = 0
        self.step_count = 0
        
        # Experience buffer (optional, for now use on-policy)
        self.use_replay_buffer = False
        if self.use_replay_buffer:
            self.replay_buffer = deque(maxlen=args.replay_buffer_size)
    
    def train_episode(self, env):
        """
        Train one episode with MCTS-guided self-play.
        
        Args:
            env: Buffer-based packing environment
            
        Returns:
            episode_stats: Dict with training statistics
        """
        # Storage for training data
        buffer_states = []
        mcts_policies = []
        rewards = []
        
        # Reset environment
        state_obs = env.reset()
        done = False
        episode_reward = 0
        
        # Self-play loop
        while not done:
            # Run MCTS to get improved policy
            with torch.no_grad():
                pi_mcts = self.mcts_planner.search(env)  # (buffer_size,)
            
            # Store state and MCTS policy for training
            buffer_states.append(state_obs.copy())
            mcts_policies.append(pi_mcts.copy())
            
            # Sample action from MCTS policy
            # Use temperature for exploration during training
            if self.args.use_temperature:
                action_idx = np.random.choice(
                    len(pi_mcts), 
                    p=pi_mcts
                )
            else:
                action_idx = np.argmax(pi_mcts)
            
            # Execute action in environment
            state_obs, reward, done, info = env.step(action_idx)
            
            rewards.append(reward)
            episode_reward += reward
            self.step_count += 1
        
        # Get final space utilization as return
        final_ratio = info.get('ratio', env.get_ratio())
        
        # Assign returns to all states
        # Simple version: use final ratio for all states
        # Advanced version: use n-step returns or MC returns
        returns = [final_ratio] * len(buffer_states)
        
        # Update networks
        train_stats = self._update_networks(
            buffer_states, 
            mcts_policies, 
            returns
        )
        
        # Update episode count
        self.episode_count += 1
        
        # Compile episode statistics
        episode_stats = {
            'episode': self.episode_count,
            'episode_reward': episode_reward,
            'space_ratio': final_ratio,
            'num_packed': info.get('counter', 0),
            'steps': len(buffer_states),
            **train_stats
        }
        
        # Log to TensorBoard
        if self.writer is not None:
            self._log_episode(episode_stats)
        
        return episode_stats
    
    def _update_networks(self, states, target_policies, returns):
        """
        Update Set Transformer and optionally PCT via supervised learning.
        
        Args:
            states: List of buffer observations
            target_policies: List of MCTS policies (soft targets)
            returns: List of returns (values)
            
        Returns:
            train_stats: Dict with loss values
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_pct_loss = 0
        n_updates = 0
        
        # Convert to tensors
        for state, pi_target, value_target in zip(states, target_policies, returns):
            # Prepare inputs
            buffer_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            pi_target_tensor = torch.FloatTensor(pi_target).to(self.device)
            value_target_tensor = torch.FloatTensor([value_target]).to(self.device)
            
            # Get mask
            mask = torch.FloatTensor(
                np.sum(state, axis=-1) == 0
            ).unsqueeze(0).to(self.device)
            
            # Forward pass through Set Transformer
            pi_pred, value_pred = self.set_transformer(buffer_tensor, mask.bool())
            pi_pred = pi_pred.squeeze(0)
            value_pred = value_pred.squeeze(0)
            
            # Policy loss: Cross-entropy between MCTS policy and predicted policy
            # KL divergence: sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
            # For training, we minimize: -sum(p_target * log(p_pred))
            policy_loss = -torch.sum(
                pi_target_tensor * torch.log(pi_pred + 1e-8)
            )
            
            # Value loss: MSE between predicted value and actual return
            value_loss = F.mse_loss(value_pred, value_target_tensor)
            
            # Combined loss for Set Transformer
            st_loss = (
                self.args.policy_loss_weight * policy_loss +
                self.args.value_loss_weight * value_loss
            )
            
            # Update Set Transformer
            self.st_optimizer.zero_grad()
            st_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.set_transformer.parameters(),
                self.args.max_grad_norm
            )
            self.st_optimizer.step()
            
            # Optionally update PCT (if training jointly)
            if self.pct_optimizer is not None and self.args.train_pct:
                # PCT can be trained with regular RL loss
                # For now, skip PCT updates (can be added later)
                pass
            
            # Accumulate statistics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_updates += 1
        
        # Return average losses
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'pct_loss': total_pct_loss / max(n_updates, 1)
        }
    
    def _log_episode(self, stats):
        """Log episode statistics to TensorBoard."""
        episode = stats['episode']
        
        self.writer.add_scalar('Episode/Reward', stats['episode_reward'], episode)
        self.writer.add_scalar('Episode/SpaceRatio', stats['space_ratio'], episode)
        self.writer.add_scalar('Episode/NumPacked', stats['num_packed'], episode)
        self.writer.add_scalar('Episode/Steps', stats['steps'], episode)
        
        self.writer.add_scalar('Loss/Policy', stats['policy_loss'], episode)
        self.writer.add_scalar('Loss/Value', stats['value_loss'], episode)
        if stats['pct_loss'] > 0:
            self.writer.add_scalar('Loss/PCT', stats['pct_loss'], episode)
    
    def train(self, env, num_episodes):
        """
        Main training loop.
        
        Args:
            env: Buffer-based environment
            num_episodes: Number of episodes to train
        """
        print(f"Starting MCTS-enhanced training for {num_episodes} episodes...")
        
        # Statistics tracking
        episode_rewards = deque(maxlen=100)
        episode_ratios = deque(maxlen=100)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Train one episode
            stats = self.train_episode(env)
            
            # Track statistics
            episode_rewards.append(stats['episode_reward'])
            episode_ratios.append(stats['space_ratio'])
            
            # Print progress
            if (episode + 1) % self.args.print_interval == 0:
                elapsed = time.time() - start_time
                fps = self.step_count / elapsed
                
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Steps: {self.step_count}, FPS: {fps:.1f}")
                print(f"  Last 100 episodes:")
                print(f"    Mean reward: {np.mean(episode_rewards):.4f}")
                print(f"    Mean ratio: {np.mean(episode_ratios):.4f}")
                print(f"    Max ratio: {np.max(episode_ratios):.4f}")
                print(f"  Losses:")
                print(f"    Policy: {stats['policy_loss']:.4f}")
                print(f"    Value: {stats['value_loss']:.4f}")
            
            # Save model
            if (episode + 1) % self.args.save_interval == 0:
                self.save_checkpoint(episode + 1)
        
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
        
        save_path = os.path.join(
            save_dir,
            f'checkpoint_ep{episode}.pt'
        )
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
