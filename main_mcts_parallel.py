"""
Main training script for MCTS-Enhanced 3D Bin Packing with PARALLEL collection.

This script uses multiprocessing to collect episodes in parallel across
multiple CPU cores, significantly improving training throughput.

Usage:
    python main_mcts_parallel.py --n-workers 4 --mcts-simulations 20
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
from model import DRL_GAT
from parallel_collector import ParallelMCTSTrainer
import givenData


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Parallel MCTS-Enhanced 3D Bin Packing'
    )
    
    # Environment settings
    parser.add_argument('--setting', type=int, default=2)
    parser.add_argument('--buffer-size', type=int, default=10)
    parser.add_argument('--total-items', type=int, default=50)  # Reduced for faster episodes
    parser.add_argument('--lnes', type=str, default='EMS')
    
    # Network architecture
    parser.add_argument('--st-d-model', type=int, default=128)
    parser.add_argument('--st-n-heads', type=int, default=4)
    parser.add_argument('--st-n-layers', type=int, default=3)
    parser.add_argument('--pct-model-path', type=str, default=None)
    
    # PCT settings
    parser.add_argument('--internal-node-holder', type=int, default=80)
    parser.add_argument('--leaf-node-holder', type=int, default=50)
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--gat-layer-num', type=int, default=1)
    
    # MCTS settings
    parser.add_argument('--mcts-simulations', type=int, default=20)  # Reduced
    parser.add_argument('--mcts-c-puct', type=float, default=1.0)
    parser.add_argument('--mcts-temperature', type=float, default=1.0)
    
    # Parallel settings
    parser.add_argument('--n-workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--episodes-per-batch', type=int, default=None,
                       help='Episodes per batch (default: n_workers)')
    
    # Training settings
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--st-learning-rate', type=float, default=1e-4)
    parser.add_argument('--use-temperature', action='store_true', default=True)
    
    # Logging
    parser.add_argument('--exp-name', type=str, default='parallel_mcts')
    parser.add_argument('--save-dir', type=str, default='./logs/parallel_experiment')
    parser.add_argument('--log-dir', type=str, default='./logs/parallel_runs')
    parser.add_argument('--print-interval', type=int, default=4)
    parser.add_argument('--save-interval', type=int, default=100)
    
    # Device
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    # Performance
    parser.add_argument('--use-amp', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Set device
    if args.no_cuda or not torch.cuda.is_available():
        args.device = torch.device('cpu')
        print("Using CPU")
    else:
        args.device = torch.device(f'cuda:{args.device}')
        print(f"Using CUDA device {args.device}")
    
    # Load settings
    args.container_size = givenData.container_size
    args.item_size_set = givenData.item_size_set
    args.normFactor = 1.0 / np.max(args.container_size)
    
    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    
    # Auto-calculate internal_node_holder
    container_volume = args.container_size[0] * args.container_size[1] * args.container_size[2]
    min_item_size = min(min(item) for item in args.item_size_set)
    max_theoretical_items = int(container_volume / (min_item_size ** 3)) + 10
    
    if args.internal_node_holder < max_theoretical_items:
        args.internal_node_holder = max_theoretical_items
    
    return args


def main():
    """Main training function."""
    args = get_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create experiment directory
    timeStr = args.exp_name + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime())
    
    # TensorBoard
    log_path = f'{args.log_dir}/{timeStr}'
    writer = SummaryWriter(logdir=log_path)
    print(f"TensorBoard logs: {log_path}")
    
    # Create networks
    print("\n" + "=" * 60)
    print("Creating Networks...")
    print("=" * 60)
    
    set_transformer = SetTransformer(
        d_model=args.st_d_model,
        n_heads=args.st_n_heads,
        n_layers=args.st_n_layers,
        buffer_size=args.buffer_size
    ).to(args.device)
    
    pct_policy = DRL_GAT(args).to(args.device)
    
    if args.pct_model_path:
        checkpoint = torch.load(args.pct_model_path, map_location=args.device)
        pct_policy.load_state_dict(checkpoint)
        print(f"Loaded PCT from {args.pct_model_path}")
    
    print(f"Set Transformer params: {sum(p.numel() for p in set_transformer.parameters())}")
    print(f"PCT params: {sum(p.numel() for p in pct_policy.parameters())}")
    
    # Create parallel trainer
    print("\n" + "=" * 60)
    print("Creating Parallel Trainer...")
    print("=" * 60)
    
    args.save_dir = f'{args.save_dir}/{timeStr}'
    
    trainer = ParallelMCTSTrainer(
        set_transformer=set_transformer,
        pct_policy=pct_policy,
        args=args,
        writer=writer,
        device=args.device,
        n_workers=args.n_workers
    )
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING PARALLEL TRAINING")
    print("=" * 60)
    
    trainer.train(
        num_episodes=args.num_episodes,
        episodes_per_batch=args.episodes_per_batch
    )
    
    writer.close()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
