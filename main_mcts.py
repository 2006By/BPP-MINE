"""
Main training script for MCTS-Enhanced Hierarchical 3D Bin Packing.

This script integrates:
1. Set Transformer for prior estimation
2. MCTS for lookahead planning  
3. PCT for item placement
4. Policy distillation training

Usage:
    python main_mcts.py --buffer-size 10 --mcts-simulations 200
"""

import sys
import torch
import time
import numpy as np
import random
import argparse
from tensorboardX import SummaryWriter

# Import components
from set_transformer import SetTransformer
from mcts_planner import MCTSPlanner
from train_mcts import MCTSTrainer
from parallel_trainer import ParallelMCTSTrainer
from model import DRL_GAT
import givenData
from pct_envs.PctDiscrete0.bin3D_buffer import PackingDiscreteWithBuffer


def get_mcts_args():
    """Parse command line arguments for MCTS training."""
    parser = argparse.ArgumentParser(
        description='MCTS-Enhanced Hierarchical 3D Bin Packing'
    )
    
    # Environment settings
    parser.add_argument('--setting', type=int, default=2,
                       help='Experiment setting (1/2/3)')
    parser.add_argument('--buffer-size', type=int, default=10,
                       help='Number of items in lookahead buffer')
    parser.add_argument('--total-items', type=int, default=150,
                       help='Total items per episode')
    parser.add_argument('--lnes', type=str, default='EMS',
                       help='Leaf Node Expansion Scheme')
    
    # Network architecture
    parser.add_argument('--st-d-model', type=int, default=128,
                       help='Set Transformer embedding dimension')
    parser.add_argument('--st-n-heads', type=int, default=4,
                       help='Set Transformer attention heads')
    parser.add_argument('--st-n-layers', type=int, default=3,
                       help='Set Transformer encoder layers')
    parser.add_argument('--pct-model-path', type=str, default=None,
                       help='Path to pre-trained PCT model')
    
    # PCT settings (reuse from original)
    parser.add_argument('--internal-node-holder', type=int, default=80)
    parser.add_argument('--leaf-node-holder', type=int, default=50)
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--gat-layer-num', type=int, default=1)
    
    # MCTS settings
    parser.add_argument('--mcts-simulations', type=int, default=200,
                       help='Number of MCTS simulations per decision')
    parser.add_argument('--mcts-c-puct', type=float, default=1.0,
                       help='Exploration constant for UCB')
    parser.add_argument('--mcts-temperature', type=float, default=1.0,
                       help='Temperature for action selection')
    
    # Training settings
    parser.add_argument('--num-episodes', type=int, default=10000,
                       help='Total training episodes')
    parser.add_argument('--st-learning-rate', type=float, default=1e-4,
                       help='Set Transformer learning rate')
    parser.add_argument('--pct-learning-rate', type=float, default=1e-5,
                       help='PCT learning rate (if training)')
    parser.add_argument('--train-pct', action='store_true',
                       help='Train PCT alongside Set Transformer')
    parser.add_argument('--policy-loss-weight', type=float, default=1.0,
                       help='Weight for policy loss')
    parser.add_argument('--value-loss-weight', type=float, default=0.5,
                       help='Weight for value loss')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Maximum gradient norm')
    parser.add_argument('--use-temperature', action='store_true', default=True,
                       help='Use temperature for action sampling')
    
    # Parallel training settings (for GPU optimization)
    parser.add_argument('--use-parallel-trainer', action='store_true',
                       help='Use parallel trainer for better GPU utilization')
    parser.add_argument('--num-parallel-envs', type=int, default=16,
                       help='Number of parallel environments (default: 16)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for network updates (default: 64)')
    
    # Logging and saving
    parser.add_argument('--exp-name', type=str, default='mcts_bin_packing',
                       help='Experiment name')
    parser.add_argument('--save-dir', type=str, default='./logs/mcts_experiment',
                       help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='./logs/mcts_runs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--print-interval', type=int, default=10,
                       help='Print statistics every N episodes')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save model every N episodes')
    
    # Device settings
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device ID')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluation mode')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint to load')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Set device
    if args.no_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.device}')
    
    # Load container and item settings from givenData
    args.container_size = givenData.container_size
    args.item_size_set = givenData.item_size_set
    
    # Compute normalization factor
    args.normFactor = 1.0 / np.max(args.container_size)
    
    # Set internal node length based on setting
    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    
    return args


def create_environment(args):
    """Create buffer-based packing environment."""
    env = PackingDiscreteWithBuffer(
        setting=args.setting,
        container_size=args.container_size,
        item_set=args.item_size_set,
        internal_node_holder=args.internal_node_holder,
        leaf_node_holder=args.leaf_node_holder,
        buffer_size=args.buffer_size,
        total_items=args.total_items,
        LNES=args.lnes
    )
    return env


def create_networks(args):
    """Create and initialize all networks."""
    
    # 1. Create Set Transformer
    set_transformer = SetTransformer(
        d_model=args.st_d_model,
        n_heads=args.st_n_heads,
        n_layers=args.st_n_layers,
        buffer_size=args.buffer_size
    ).to(args.device)
    
    print(f"Created Set Transformer with {sum(p.numel() for p in set_transformer.parameters())} parameters")
    
    # 2. Create or load PCT policy
    pct_policy = DRL_GAT(args).to(args.device)
    
    if args.pct_model_path is not None:
        # Load pre-trained PCT
        print(f"Loading pre-trained PCT from {args.pct_model_path}")
        checkpoint = torch.load(args.pct_model_path, map_location=args.device)
        pct_policy.load_state_dict(checkpoint)
        print("PCT loaded successfully")
    else:
        print("Training PCT from scratch")
    
    print(f"Created PCT policy with {sum(p.numel() for p in pct_policy.parameters())} parameters")
    
    return set_transformer, pct_policy


def main(args):
    """Main training function."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # Create experiment directory
    timeStr = args.exp_name + '-' + time.strftime(
        '%Y.%m.%d-%H-%M-%S', 
        time.localtime(time.time())
    )
    
    # Setup TensorBoard
    log_path = f'{args.log_dir}/{timeStr}'
    writer = SummaryWriter(logdir=log_path)
    print(f"TensorBoard logs: {log_path}")
    
    # Create environment
    print("\n" + "="*60)
    print("Creating Environment...")
    print("="*60)
    env = create_environment(args)
    print(f"✓ Environment created")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Total items per episode: {args.total_items}")
    print(f"  Container size: {args.container_size}")
    
    # Create networks
    print("\n" + "="*60)
    print("Creating Networks...")
    print("="*60)
    set_transformer, pct_policy = create_networks(args)
    print(f"✓ Networks created")
    
    # Create MCTS planner
    print("\n" + "="*60)
    print("Creating MCTS Planner...")
    print("="*60)
    mcts_planner = MCTSPlanner(
        set_transformer=set_transformer,
        pct_policy=pct_policy,
        n_simulations=args.mcts_simulations,
        c_puct=args.mcts_c_puct,
        temperature=args.mcts_temperature,
        device=args.device
    )
    print(f"✓ MCTS Planner created")
    print(f"  Simulations per decision: {args.mcts_simulations}")
    print(f"  Exploration constant (c_puct): {args.mcts_c_puct}")
    
    # Create trainer
    print("\n" + "="*60)
    print("Creating Trainer...")
    print("="*60)
    
    args.save_dir = f'{args.save_dir}/{timeStr}'
    
    # Choose trainer based on parallel option
    if args.use_parallel_trainer:
        trainer = ParallelMCTSTrainer(
            set_transformer=set_transformer,
            pct_policy=pct_policy,
            mcts_planner=mcts_planner,
            args=args,
            writer=writer,
            device=args.device,
            num_parallel_envs=args.num_parallel_envs
        )
        print(f"✓ Parallel Trainer created (GPU-optimized)")
        print(f"  Parallel environments: {args.num_parallel_envs}")
        print(f"  Batch size: {args.batch_size}")
    else:
        trainer = MCTSTrainer(
            set_transformer=set_transformer,
            pct_policy=pct_policy,
            mcts_planner=mcts_planner,
            args=args,
            writer=writer,
            device=args.device
        )
        print(f"✓ Standard Trainer created")
    
    print(f"  Set Transformer LR: {args.st_learning_rate}")
    print(f"  Train PCT: {args.train_pct}")
    
    # Load checkpoint if specified
    if args.load_checkpoint is not None:
        print(f"\nLoading checkpoint from {args.load_checkpoint}")
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Training or evaluation
    if args.evaluate:
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)
        evaluate(trainer, env, args.eval_episodes)
    else:
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Total episodes: {args.num_episodes}")
        print(f"Save interval: {args.save_interval} episodes")
        print(f"Print interval: {args.print_interval} episodes")
        print("="*60 + "\n")
        
        # Start training
        trainer.train(env, args.num_episodes)
    
    # Close writer
    writer.close()
    print("\nTraining complete!")


def evaluate(trainer, env, num_episodes):
    """Evaluate trained model."""
    print(f"Evaluating for {num_episodes} episodes...")
    
    ratios = []
    num_packed_list = []
    
    trainer.set_transformer.eval()
    trainer.pct_policy.eval()
    
    for ep in range(num_episodes):
        state_obs = env.reset()
        done = False
        
        while not done:
            # Use MCTS for evaluation (can also use Set Transformer directly)
            pi_mcts = trainer.mcts_planner.search(env)
            action_idx = np.argmax(pi_mcts)
            
            state_obs, reward, done, info = env.step(action_idx)
        
        ratio = info.get('ratio', env.get_ratio())
        num_packed = info.get('counter', 0)
        
        ratios.append(ratio)
        num_packed_list.append(num_packed)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}: ratio={ratio:.4f}, packed={num_packed}")
    
    print(f"\nEvaluation Results:")
    print(f"  Mean ratio: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
    print(f"  Max ratio: {np.max(ratios):.4f}")
    print(f"  Min ratio: {np.min(ratios):.4f}")
    print(f"  Mean items packed: {np.mean(num_packed_list):.1f}")


if __name__ == '__main__':
    args = get_mcts_args()
    main(args)
