import torch
import numpy as np
import math
import copy
from typing import Optional, Dict, Tuple, List

class MCTSNode:
    """
    Node in the Monte Carlo Search Tree.
    
    Each node represents a state in the search process.
    For bin packing, a state consists of:
    - Current bin configuration
    - Remaining items in buffer
    """
    
    def __init__(self, 
                 state,
                 parent: Optional['MCTSNode'] = None,
                 prior: float = 0.0,
                 action_idx: Optional[int] = None):
        """
        Args:
            state: Environment state (will be copied for simulation)
            parent: Parent node in the tree
            prior: Prior probability P(a|s) from Set Transformer
            action_idx: Action (buffer index) that led to this node
        """
        self.state = state
        self.parent = parent
        self.prior = prior
        self.action_idx = action_idx
        
        # MCTS statistics
        self.children: Dict[int, 'MCTSNode'] = {}  # action_idx -> child node
        self.visit_count = 0
        self.total_value = 0.0  # Accumulated Q-value
        
        # Terminal flag
        self._is_terminal = None
        
    @property
    def Q(self) -> float:
        """Mean action-value (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (buffer empty or packing failed)."""
        if self._is_terminal is None:
            # Cache terminal status
            self._is_terminal = (
                len(self.state.buffer) == 0 or 
                self.state.done
            )
        return self._is_terminal
    
    def is_expanded(self) -> bool:
        """Check if this node has been expanded."""
        return len(self.children) > 0
    
    def ucb_score(self, c_puct: float = 1.0, parent_visit: int = 1) -> float:
        """
        Upper Confidence Bound for Trees (UCT).
        
        Formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: Exploration constant (higher = more exploration)
            parent_visit: Parent's visit count
            
        Returns:
            UCB score
        """
        # Exploitation term: mean Q-value
        q_value = self.Q
        
        # Exploration term: based on prior and visit counts
        u_value = c_puct * self.prior * math.sqrt(parent_visit) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """
        Select child with highest UCB score.
        
        Returns:
            Child node with max UCB
        """
        best_child = None
        best_score = -float('inf')
        
        for child in self.children.values():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def add_child(self, action_idx: int, child_state, prior: float) -> 'MCTSNode':
        """
        Add a child node.
        
        Args:
            action_idx: Buffer index of selected item
            child_state: Resulting state after action
            prior: Prior probability from Set Transformer
            
        Returns:
            Created child node
        """
        child = MCTSNode(
            state=child_state,
            parent=self,
            prior=prior,
            action_idx=action_idx
        )
        self.children[action_idx] = child
        return child
    
    def update(self, value: float):
        """
        Update statistics after backpropagation.
        
        Args:
            value: Value to add (typically space utilization ratio)
        """
        self.visit_count += 1
        self.total_value += value
    
    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities based on visit counts.
        
        Args:
            temperature: Temperature for softmax
                        - temperature = 0: deterministic (argmax)
                        - temperature = 1: proportional to visit count
                        - temperature > 1: more exploration
                        
        Returns:
            Probability distribution over actions
        """
        if not self.children:
            return None
            
        action_indices = []
        visit_counts = []
        
        for action_idx, child in self.children.items():
            action_indices.append(action_idx)
            visit_counts.append(child.visit_count)
            
        visit_counts = np.array(visit_counts, dtype=np.float32)
        
        if temperature == 0:
            # Deterministic: select most visited
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            # Softmax with temperature
            visit_counts = visit_counts ** (1.0 / temperature)
            probs = visit_counts / visit_counts.sum()
            
        # Create full probability vector
        buffer_size = self.state.buffer_size
        action_probs = np.zeros(buffer_size, dtype=np.float32)
        for idx, action_idx in enumerate(action_indices):
            action_probs[action_idx] = probs[idx]
            
        return action_probs


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner for lookahead planning.
    
    Uses Set Transformer for prior and PCT for simulation.
    """
    
    def __init__(self,
                 set_transformer,
                 pct_policy,
                 n_simulations: int = 200,
                 c_puct: float = 1.0,
                 temperature: float = 1.0,
                 device='cuda'):
        """
        Args:
            set_transformer: Set Transformer network for priors
            pct_policy: PCT network for item placement
            n_simulations: Number of MCTS simulations per search
            c_puct: Exploration constant for UCB
            temperature: Temperature for final action selection
            device: Device to run networks on
        """
        self.set_transformer = set_transformer
        self.pct_policy = pct_policy
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device
        
    def search(self, root_state) -> np.ndarray:
        """
        Run MCTS search from root state.
        
        Args:
            root_state: Initial environment state
            
        Returns:
            action_probs: Probability distribution over buffer items (size: buffer_size)
        """
        # Create root node
        root = MCTSNode(state=root_state)
        
        # Run simulations
        for _ in range(self.n_simulations):
            # 1. Selection: Traverse tree with UCB
            node = self._select(root)
            
            # 2. Expansion: Expand leaf node if not terminal
            if not node.is_terminal() and node.visit_count > 0:
                node = self._expand(node)
            
            # 3. Simulation: Rollout with PCT to get value
            value = self._simulate(node)
            
            # 4. Backpropagation: Update all ancestors
            self._backpropagate(node, value)
        
        # Get action probabilities from visit counts
        action_probs = root.get_action_probs(temperature=self.temperature)
        
        return action_probs
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Traverse tree using UCB until reaching leaf.
        
        Args:
            node: Starting node (usually root)
            
        Returns:
            Leaf node
        """
        while node.is_expanded() and not node.is_terminal():
            node = node.select_child(c_puct=self.c_puct)
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add children with priors from Set Transformer.
        
        Args:
            node: Node to expand
            
        Returns:
            One of the newly created children
        """
        # Get prior probabilities from Set Transformer
        buffer_items = self._get_buffer_tensor(node.state)
        mask = self._get_buffer_mask(node.state)
        
        with torch.no_grad():
            priors, _ = self.set_transformer(buffer_items, mask)
            priors = priors.cpu().numpy()[0]  # Shape: (buffer_size,)
        
        # Create child for each valid buffer item
        valid_actions = [i for i in range(len(node.state.buffer))]
        
        for action_idx in valid_actions:
            # Copy state for simulation
            child_state = copy.deepcopy(node.state)
            
            # Simulate action (will be executed in _simulate)
            # Here we just create the node structure
            node.add_child(action_idx, child_state, priors[action_idx])
        
        # Return first child for further simulation
        # (In practice, could select by UCB here too)
        if node.children:
            return next(iter(node.children.values()))
        return node
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: Rollout using PCT to get final value.
        
        This is the key innovation: use PCT as deterministic transition model.
        
        Args:
            node: Node to simulate from
            
        Returns:
            value: Final space utilization ratio
        """
        # Copy state for simulation
        state = copy.deepcopy(node.state)
        
        # If this is a newly expanded child, execute its action first
        if node.action_idx is not None and not hasattr(state, '_action_executed'):
            state = self._execute_action(state, node.action_idx)
            state._action_executed = True
        
        # Rollout until buffer empty or failure
        while not state.done and len(state.buffer) > 0:
            # Use Set Transformer to select next item (greedy)
            buffer_items = self._get_buffer_tensor(state)
            mask = self._get_buffer_mask(state)
            
            with torch.no_grad():
                probs, _ = self.set_transformer(buffer_items, mask)
                action_idx = probs.argmax(dim=-1).item()
            
            # Execute action
            state = self._execute_action(state, action_idx)
        
        # Return final space utilization as value
        value = state.get_ratio()
        return value
    
    def _execute_action(self, state, action_idx):
        """
        Execute action using PCT for placement.
        
        Args:
            state: Current environment state
            action_idx: Index of item in buffer to place
            
        Returns:
            next_state: State after placement
        """
        # Get selected item from buffer
        selected_item = state.buffer[action_idx]
        
        # Use PCT to find best placement position
        pct_obs = state.get_pct_observation(selected_item)
        # Reshape from 1D to 3D: (batch=1, num_nodes, node_features=9)
        # Observation structure: internal_nodes + leaf_nodes + next_item, each with 9 features
        total_nodes = state.internal_node_holder + state.leaf_node_holder + state.next_holder
        pct_obs_tensor = torch.FloatTensor(pct_obs).reshape(1, total_nodes, 9).to(self.device)
        
        with torch.no_grad():
            # PCT outputs leaf node index
            _, leaf_idx, _, _ = self.pct_policy(pct_obs_tensor, deterministic=True)
            leaf_idx = leaf_idx.item()
        
        # Execute action in environment
        next_state, reward, done, _ = state.step(action_idx, leaf_idx)
        
        return next_state
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagation phase: Update all ancestors with value.
        
        Args:
            node: Leaf node where simulation ended
            value: Value to propagate
        """
        while node is not None:
            node.update(value)
            node = node.parent
    
    def _get_buffer_tensor(self, state) -> torch.Tensor:
        """
        Convert buffer to tensor for Set Transformer.
        
        Returns:
            Tensor of shape (1, buffer_size, 4)
        """
        buffer_items = state.get_buffer_features()
        return torch.FloatTensor(buffer_items).unsqueeze(0).to(self.device)
    
    def _get_buffer_mask(self, state) -> Optional[torch.Tensor]:
        """
        Get mask for invalid buffer positions.
        
        Returns:
            Mask of shape (1, buffer_size) or None
        """
        if hasattr(state, 'get_buffer_mask'):
            mask = state.get_buffer_mask()
            return torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        return None


def test_mcts():
    """Simple test for MCTS planner (requires environment to be implemented)."""
    print("MCTS Planner structure created successfully!")
    print("\nKey components:")
    print("- MCTSNode: Tree node with UCB scoring")
    print("- MCTSPlanner: Main search algorithm")
    print("- Integration points: Set Transformer (prior) + PCT (simulation)")
    
    # Test node creation
    class DummyState:
        def __init__(self):
            self.buffer = [1, 2, 3]
            self.done = False
            self.buffer_size = 10
    
    state = DummyState()
    node = MCTSNode(state=state, prior=0.5)
    
    print(f"\n✓ Created MCTS node with UCB = {node.ucb_score():.4f}")
    
    # Test UCB calculation
    node.update(0.7)
    node.update(0.8)
    print(f"✓ After 2 visits: Q = {node.Q:.4f}, visits = {node.visit_count}")

if __name__ == '__main__':
    test_mcts()
