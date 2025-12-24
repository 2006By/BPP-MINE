"""
Parallel data collector for MCTS training using multiprocessing.

This module enables parallel episode collection across multiple CPU cores,
significantly improving training throughput for CPU-bound MCTS operations.
"""

import torch
import torch.multiprocessing as mp
import numpy as np
from collections import deque
import time
import copy

# Set multiprocessing start method for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


def worker_collect_episode(args):
    """
    Worker function to collect one episode of experience.
    
    This function runs in a separate process and does NOT use GPU.
    It only collects (state, action, reward) tuples using a simple policy.
    
    Args:
        args: Tuple of (worker_id, env_config, policy_weights, mcts_config)
        
    Returns:
        dict: Episode experience data
    """
    worker_id, env_config, policy_weights, mcts_config = args
    
    # Import here to avoid issues with multiprocessing
    from pct_envs.PctDiscrete0.bin3D_buffer import PackingDiscreteWithBuffer
    from set_transformer import SetTransformer
    from mcts_planner import MCTSPlanner, MCTSNode
    from model import DRL_GAT
    
    # Create environment for this worker
    env = PackingDiscreteWithBuffer(
        setting=env_config['setting'],
        container_size=env_config['container_size'],
        item_set=env_config['item_set'],
        internal_node_holder=env_config['internal_node_holder'],
        leaf_node_holder=env_config['leaf_node_holder'],
        buffer_size=env_config['buffer_size'],
        total_items=env_config['total_items'],
        LNES=env_config['lnes']
    )
    
    # Create networks on CPU for this worker
    device = torch.device('cpu')
    
    set_transformer = SetTransformer(
        d_model=policy_weights['st_d_model'],
        n_heads=policy_weights['st_n_heads'],
        n_layers=policy_weights['st_n_layers'],
        buffer_size=env_config['buffer_size']
    ).to(device)
    set_transformer.load_state_dict(policy_weights['set_transformer'])
    set_transformer.eval()
    
    pct_policy = DRL_GAT(policy_weights['pct_args']).to(device)
    pct_policy.load_state_dict(policy_weights['pct_policy'])
    pct_policy.eval()
    
    # Create lightweight MCTS planner (fewer simulations for workers)
    mcts_planner = MCTSPlanner(
        set_transformer=set_transformer,
        pct_policy=pct_policy,
        n_simulations=mcts_config['n_simulations'],
        c_puct=mcts_config['c_puct'],
        temperature=mcts_config['temperature'],
        device=device,
        batch_size=1  # No batching on CPU
    )
    
    # Collect episode
    buffer_states = []
    mcts_policies = []
    rewards = []
    
    state_obs = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Run MCTS search
        with torch.no_grad():
            pi_mcts = mcts_planner.search(env)
        
        buffer_states.append(state_obs.copy())
        mcts_policies.append(pi_mcts.copy())
        
        # Sample action
        if mcts_config['use_temperature']:
            action_idx = np.random.choice(len(pi_mcts), p=pi_mcts)
        else:
            action_idx = np.argmax(pi_mcts)
        
        state_obs, reward, done, info = env.step(action_idx)
        rewards.append(reward)
        episode_reward += reward
    
    final_ratio = info.get('ratio', env.get_ratio())
    
    return {
        'worker_id': worker_id,
        'buffer_states': buffer_states,
        'mcts_policies': mcts_policies,
        'rewards': rewards,
        'episode_reward': episode_reward,
        'final_ratio': final_ratio,
        'num_packed': info.get('counter', 0),
        'steps': len(buffer_states)
    }


class ParallelCollector:
    """
    Parallel episode collector using multiprocessing.
    
    Spawns multiple worker processes to collect episodes in parallel,
    then aggregates the experiences for batch training on GPU.
    """
    
    def __init__(self, 
                 set_transformer,
                 pct_policy,
                 args,
                 n_workers=4):
        """
        Args:
            set_transformer: Set Transformer network (for weights)
            pct_policy: PCT network (for weights)
            args: Training arguments
            n_workers: Number of parallel workers
        """
        self.set_transformer = set_transformer
        self.pct_policy = pct_policy
        self.args = args
        self.n_workers = n_workers
        
        # Environment config (serializable)
        self.env_config = {
            'setting': args.setting,
            'container_size': args.container_size,
            'item_set': args.item_size_set,
            'internal_node_holder': args.internal_node_holder,
            'leaf_node_holder': args.leaf_node_holder,
            'buffer_size': args.buffer_size,
            'total_items': args.total_items,
            'lnes': args.lnes
        }
        
        # MCTS config
        self.mcts_config = {
            'n_simulations': max(args.mcts_simulations // 2, 10),  # Fewer for workers
            'c_puct': args.mcts_c_puct,
            'temperature': args.mcts_temperature,
            'use_temperature': args.use_temperature
        }
        
        print(f"ParallelCollector initialized with {n_workers} workers")
        print(f"  Worker MCTS simulations: {self.mcts_config['n_simulations']}")
    
    def _get_policy_weights(self):
        """Get serializable policy weights for workers."""
        return {
            'set_transformer': {k: v.cpu() for k, v in self.set_transformer.state_dict().items()},
            'pct_policy': {k: v.cpu() for k, v in self.pct_policy.state_dict().items()},
            'st_d_model': self.args.st_d_model,
            'st_n_heads': self.args.st_n_heads,
            'st_n_layers': self.args.st_n_layers,
            'pct_args': self.args
        }
    
    def collect_batch(self, n_episodes=None):
        """
        Collect multiple episodes in parallel.
        
        Args:
            n_episodes: Number of episodes to collect (default: n_workers)
            
        Returns:
            list: List of episode experience dicts
        """
        if n_episodes is None:
            n_episodes = self.n_workers
        
        policy_weights = self._get_policy_weights()
        
        # Prepare worker arguments
        worker_args = [
            (i, self.env_config, policy_weights, self.mcts_config)
            for i in range(n_episodes)
        ]
        
        # Run workers in parallel
        start_time = time.time()
        
        with mp.Pool(processes=min(n_episodes, self.n_workers)) as pool:
            results = pool.map(worker_collect_episode, worker_args)
        
        collect_time = time.time() - start_time
        
        # Statistics
        total_steps = sum(r['steps'] for r in results)
        avg_ratio = np.mean([r['final_ratio'] for r in results])
        
        print(f"  Collected {n_episodes} episodes in {collect_time:.2f}s")
        print(f"  Total steps: {total_steps}, Avg ratio: {avg_ratio:.4f}")
        print(f"  Throughput: {total_steps / collect_time:.1f} steps/s")
        
        return results


class ParallelMCTSTrainer:
    """
    MCTS Trainer with parallel episode collection.
    
    Uses multiple CPU workers to collect episodes in parallel,
    then performs batch training updates on GPU.
    """
    
    def __init__(self,
                 set_transformer,
                 pct_policy,
                 args,
                 writer=None,
                 device='cuda',
                 n_workers=4):
        """
        Args:
            set_transformer: Set Transformer network
            pct_policy: PCT network
            args: Training arguments
            writer: TensorBoard writer
            device: Device for training (GPU)
            n_workers: Number of parallel workers
        """
        self.set_transformer = set_transformer
        self.pct_policy = pct_policy
        self.args = args
        self.writer = writer
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            set_transformer.parameters(),
            lr=args.st_learning_rate
        )
        
        # Parallel collector
        self.collector = ParallelCollector(
            set_transformer=set_transformer,
            pct_policy=pct_policy,
            args=args,
            n_workers=n_workers
        )
        
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        
        # Mixed precision
        self.use_amp = getattr(args, 'use_amp', True) and device.type == 'cuda'
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
    
    def train(self, num_episodes, episodes_per_batch=None):
        """
        Main training loop with parallel collection.
        
        Args:
            num_episodes: Total episodes to train
            episodes_per_batch: Episodes to collect per batch (default: n_workers)
        """
        if episodes_per_batch is None:
            episodes_per_batch = self.collector.n_workers
        
        print(f"\nStarting parallel MCTS training")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Episodes per batch: {episodes_per_batch}")
        print(f"  Workers: {self.collector.n_workers}")
        print("=" * 60)
        
        episode_ratios = deque(maxlen=100)
        start_time = time.time()
        
        while self.episode_count < num_episodes:
            batch_start = time.time()
            
            # Collect episodes in parallel
            experiences = self.collector.collect_batch(episodes_per_batch)
            
            # Aggregate all experiences
            all_states = []
            all_policies = []
            all_returns = []
            
            for exp in experiences:
                all_states.extend(exp['buffer_states'])
                all_policies.extend(exp['mcts_policies'])
                # Use final ratio as return for all states in episode
                all_returns.extend([exp['final_ratio']] * len(exp['buffer_states']))
                
                episode_ratios.append(exp['final_ratio'])
                self.step_count += exp['steps']
            
            # Batch training update on GPU
            train_stats = self._update_networks(all_states, all_policies, all_returns)
            
            self.episode_count += len(experiences)
            batch_time = time.time() - batch_start
            
            # Print progress
            if self.episode_count % self.args.print_interval == 0 or self.episode_count <= episodes_per_batch:
                elapsed = time.time() - start_time
                fps = self.step_count / elapsed
                
                print(f"\nEpisode {self.episode_count}/{num_episodes}")
                print(f"  Time: {elapsed:.1f}s, Batch time: {batch_time:.2f}s")
                print(f"  Steps: {self.step_count}, FPS: {fps:.1f}")
                print(f"  Last 100 episodes:")
                print(f"    Mean ratio: {np.mean(episode_ratios):.4f}")
                print(f"    Max ratio: {np.max(episode_ratios):.4f}")
                print(f"  Losses:")
                print(f"    Policy: {train_stats['policy_loss']:.4f}")
                print(f"    Value: {train_stats['value_loss']:.4f}")
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Episode/MeanRatio', np.mean(episode_ratios), self.episode_count)
                    self.writer.add_scalar('Performance/FPS', fps, self.episode_count)
                    self.writer.add_scalar('Loss/Policy', train_stats['policy_loss'], self.episode_count)
                    self.writer.add_scalar('Loss/Value', train_stats['value_loss'], self.episode_count)
            
            # Save checkpoint
            if self.episode_count % self.args.save_interval == 0:
                self._save_checkpoint()
        
        print(f"\nTraining complete!")
        print(f"  Total time: {time.time() - start_time:.1f}s")
        print(f"  Final FPS: {self.step_count / (time.time() - start_time):.1f}")
        print(f"  Final mean ratio: {np.mean(episode_ratios):.4f}")
    
    def _update_networks(self, states, policies, returns):
        """Batch update networks on GPU."""
        if len(states) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Convert to tensors
        batch_states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
        batch_policies = torch.stack([torch.FloatTensor(p) for p in policies]).to(self.device)
        batch_returns = torch.FloatTensor(returns).to(self.device)
        
        # Mask for padding
        batch_masks = torch.sum(batch_states, dim=-1) == 0
        
        # Forward pass
        if self.use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                pi_pred, value_pred = self.set_transformer(batch_states, batch_masks.bool())
                value_pred = value_pred.squeeze(-1)
                
                policy_loss = -torch.sum(batch_policies * torch.log(pi_pred + 1e-8), dim=-1).mean()
                value_loss = torch.nn.functional.mse_loss(value_pred, batch_returns)
                
                total_loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.set_transformer.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pi_pred, value_pred = self.set_transformer(batch_states, batch_masks.bool())
            value_pred = value_pred.squeeze(-1)
            
            policy_loss = -torch.sum(batch_policies * torch.log(pi_pred + 1e-8), dim=-1).mean()
            value_loss = torch.nn.functional.mse_loss(value_pred, batch_returns)
            
            total_loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.set_transformer.parameters(), 0.5)
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        import os
        save_dir = self.args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        checkpoint = {
            'episode': self.episode_count,
            'set_transformer': self.set_transformer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
        save_path = os.path.join(save_dir, f'checkpoint_ep{self.episode_count}.pt')
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
