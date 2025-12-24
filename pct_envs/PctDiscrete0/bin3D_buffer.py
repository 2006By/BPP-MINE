from .space import Space
import numpy as np
import gym
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator
import torch
import random
import copy

class PackingDiscreteWithBuffer(gym.Env):
    """
    Extended 3D bin packing environment with buffer mechanism.
    
    Key differences from original PackingDiscrete:
    1. Maintains a buffer of N items (default N=10)
    2. Action selects item from buffer (not just placement)
    3. Buffer refills from item queue
    4. Designed for hierarchical RL with Set Transformer + PCT
    """
    
    def __init__(self,
                 setting,
                 container_size=(10, 10, 10),
                 item_set=None, 
                 data_name=None, 
                 load_test_data=False,
                 internal_node_holder=80, 
                 leaf_node_holder=50, 
                 next_holder=1, 
                 shuffle=False,
                 buffer_size=10,
                 total_items=150,
                 LNES='EMS',
                 **kwargs):
        """
        Args:
            buffer_size: Number of items in lookahead buffer (N=10)
            total_items: Total items in the item queue per episode
            Other args same as original PackingDiscrete
        """
        
        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = next_holder
        self.buffer_size = buffer_size
        self.total_items = total_items
        
        self.shuffle = shuffle
        self.bin_size = container_size
        self.size_minimum = np.min(np.array(item_set))
        self.setting = setting
        self.item_set = item_set
        if self.setting == 2: 
            self.orientation = 6
        else: 
            self.orientation = 2
        
        # The class that maintains the contents of the bin
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder)
        
        # Generator for train/test data
        if not load_test_data:
            assert item_set is not None
            self.box_creator = RandomBoxCreator(item_set)
            assert isinstance(self.box_creator, BoxCreator)
        if load_test_data:
            self.box_creator = LoadBoxCreator(data_name)
        
        self.test = load_test_data
        
        # Observation space: buffer items (buffer_size x 4)
        # Each item: (L, W, H, Density)
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=max(self.bin_size),
            shape=(self.buffer_size, 4)
        )
        
        # Action space: (buffer_index, leaf_node_index)
        # But during MCTS, we only select buffer index
        # PCT handles leaf node selection
        
        self.LNES = LNES
        
        # Buffer and queue
        self.buffer = []  # Current buffer of N items
        self.item_queue = []  # Remaining items to pack
        self.done = False
        
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]
    
    def reset(self):
        """Reset environment and initialize buffer."""
        self.box_creator.reset()
        self.packed = []
        self.space.reset()
        self.done = False
        
        # Generate full item sequence
        self.item_queue = []
        for _ in range(self.total_items):
            self.box_creator.generate_box_size()
            item = self.box_creator.preview(1)[0]
            self.item_queue.append(self._process_item(item))
        
        # Fill initial buffer
        self.buffer = []
        self._refill_buffer()
        
        return self._get_buffer_observation()
    
    def _process_item(self, item):
        """Process item to standard format."""
        if self.test:
            if self.setting == 3:
                density = item[3]
            else:
                density = 1.0
            size = [int(item[0]), int(item[1]), int(item[2])]
        else:
            if self.setting < 3:
                density = 1.0
            else:
                density = np.random.random()
                while density == 0:
                    density = np.random.random()
            size = [item[0], item[1], item[2]]
        
        return {
            'size': size,
            'density': density,
            'original': item
        }
    
    def _refill_buffer(self):
        """Refill buffer from queue to maintain buffer_size items."""
        while len(self.buffer) < self.buffer_size and len(self.item_queue) > 0:
            self.buffer.append(self.item_queue.pop(0))
    
    def step(self, action):
        """
        Execute action: select item from buffer and place it.
        
        Args:
            action: Can be:
                - int: buffer index (PCT will be called to find placement)
                - tuple: (buffer_idx, leaf_node_idx) for direct control
                
        Returns:
            observation, reward, done, info
        """
        # Parse action
        if isinstance(action, (tuple, list, np.ndarray)):
            buffer_idx = int(action[0])
            leaf_node_idx = int(action[1]) if len(action) > 1 else None
        else:
            buffer_idx = int(action)
            leaf_node_idx = None
        
        # Validate buffer index
        if buffer_idx >= len(self.buffer) or buffer_idx < 0:
            # Invalid action
            self.done = True
            return self._get_buffer_observation(), 0.0, True, {
                'counter': len(self.space.boxes),
                'ratio': self.space.get_ratio(),
                'error': 'invalid_buffer_index'
            }
        
        # Get selected item from buffer
        selected_item = self.buffer.pop(buffer_idx)
        next_box = selected_item['size']
        next_den = selected_item['density']
        
        # If no leaf node specified, we need PCT to determine placement
        # For now, we'll use a simple placement strategy
        # In actual use, this should call PCT network
        if leaf_node_idx is None:
            # This will be handled by MCTS/PCT in actual training
            # For standalone use, find first valid position
            leaf_node_idx = self._find_valid_placement(next_box, next_den)
            
            if leaf_node_idx is None:
                # No valid placement found
                self.done = True
                return self._get_buffer_observation(), 0.0, True, {
                    'counter': len(self.space.boxes),
                    'ratio': self.space.get_ratio(),
                    'error': 'no_valid_placement'
                }
        
        # Execute placement
        succeeded = self._place_item(next_box, leaf_node_idx, next_den)
        
        if not succeeded:
            self.done = True
            reward = 0.0
            info = {
                'counter': len(self.space.boxes),
                'ratio': self.space.get_ratio(),
                'reward': self.space.get_ratio() * 10
            }
            return self._get_buffer_observation(), reward, True, info
        
        # Refill buffer from queue
        self._refill_buffer()
        
        # Calculate reward (incremental space utilization)
        box_ratio = self._get_box_ratio(next_box)
        reward = box_ratio * 10
        
        # Check termination
        done = (len(self.buffer) == 0 and len(self.item_queue) == 0)
        self.done = done
        
        info = {
            'counter': len(self.space.boxes),
            'ratio': self.space.get_ratio() if done else None
        }
        
        return self._get_buffer_observation(), reward, done, info
    
    def _find_valid_placement(self, box_size, density):
        """Find first valid placement position (fallback when PCT not available)."""
        # Generate possible positions
        if self.LNES == 'EMS':
            all_positions = self.space.EMSPoint(box_size, self.setting)
        elif self.LNES == 'EV':
            all_positions = self.space.EventPoint(box_size, self.setting)
        else:
            all_positions = []
        
        # Try each position
        for pos in all_positions:
            xs, ys, zs, xe, ye, ze = pos
            x, y, z = xe - xs, ye - ys, ze - zs
            
            if self.space.drop_box_virtual([x, y, z], (xs, ys), False, density, self.setting):
                return pos
        
        return None
    
    def _place_item(self, box_size, leaf_node_placement, density):
        """
        Place item using leaf node information.
        
        Args:
            box_size: [x, y, z] dimensions
            leaf_node_placement: Position info or index
            density: Item density
            
        Returns:
            success: Whether placement succeeded
        """
        if isinstance(leaf_node_placement, (tuple, list, np.ndarray)) and len(leaf_node_placement) >= 6:
            # Direct placement coordinates
            xs, ys, zs, xe, ye, ze = leaf_node_placement[:6]
            x, y, z = xe - xs, ye - ys, ze - zs
            
            # Determine rotation
            flag = 0
            if abs(x - box_size[1]) < 1e-6 and abs(y - box_size[0]) < 1e-6:
                flag = 1
            
            succeeded = self.space.drop_box(box_size, (xs, ys), flag, density, self.setting)
        else:
            # Simplified: assume no rotation
            succeeded = False
        
        if succeeded:
            packed_box = self.space.boxes[-1]
            self.packed.append([
                packed_box.x, packed_box.y, packed_box.z,
                packed_box.lx, packed_box.ly, packed_box.lz, 0
            ])
            
            # Update EMS
            if self.LNES == 'EMS':
                self.space.GENEMS([
                    packed_box.lx, packed_box.ly, packed_box.lz,
                    packed_box.lx + packed_box.x,
                    packed_box.ly + packed_box.y,
                    packed_box.lz + packed_box.z
                ])
        
        return succeeded
    
    def _get_box_ratio(self, box_size):
        """Calculate volume ratio of box to container."""
        box_volume = box_size[0] * box_size[1] * box_size[2]
        container_volume = self.bin_size[0] * self.bin_size[1] * self.bin_size[2]
        return box_volume / container_volume
    
    def _get_buffer_observation(self):
        """
        Get observation for Set Transformer.
        
        Returns:
            np.array of shape (buffer_size, 4): [L, W, H, Density] for each item
        """
        obs = np.zeros((self.buffer_size, 4), dtype=np.float32)
        
        for i, item in enumerate(self.buffer):
            if i < self.buffer_size:
                size = item['size']
                density = item['density']
                obs[i] = [size[0], size[1], size[2], density]
        
        return obs
    
    def get_buffer_features(self):
        """Alias for _get_buffer_observation() for MCTS compatibility."""
        return self._get_buffer_observation()
    
    def get_buffer_mask(self):
        """
        Get mask for invalid buffer positions.
        
        Returns:
            Boolean mask of shape (buffer_size,): True for invalid positions
        """
        mask = np.ones(self.buffer_size, dtype=bool)
        mask[:len(self.buffer)] = False  # Valid items
        return mask
    
    def get_pct_observation(self, selected_item):
        """
        Get observation for PCT network to determine placement.
        
        Args:
            selected_item: Item dict with 'size' and 'density'
            
        Returns:
            PCT observation in original format
        """
        # This needs to match original PCT observation format
        # (internal_nodes + leaf_nodes + next_item)
        box_vec = self.space.box_vec  # Internal nodes
        
        # Get possible leaf nodes for this item
        next_box = selected_item['size']
        leaf_nodes = self._get_possible_positions(next_box, selected_item['density'])
        
        # Next item vector
        next_box_sorted = sorted(next_box)
        next_item_vec = np.zeros((1, 9))
        next_item_vec[:, 3:6] = next_box_sorted
        next_item_vec[:, 0] = selected_item['density']
        next_item_vec[:, -1] = 1
        
        # Concatenate
        obs = np.concatenate([box_vec, leaf_nodes, next_item_vec])
        return obs.reshape(-1)
    
    def _get_possible_positions(self, next_box, density):
        """Get possible positions (same as original PCT)."""
        if self.LNES == 'EMS':
            all_position = self.space.EMSPoint(next_box, self.setting)
        elif self.LNES == 'EV':
            all_position = self.space.EventPoint(next_box, self.setting)
        else:
            all_position = []
        
        if self.shuffle:
            np.random.shuffle(all_position)
        
        leaf_node_idx = 0
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []
        
        for position in all_position:
            xs, ys, zs, xe, ye, ze = position
            x, y, z = xe - xs, ye - ys, ze - zs
            
            if self.space.drop_box_virtual([x, y, z], (xs, ys), False, density, self.setting):
                tmp_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])
                leaf_node_idx += 1
            
            if leaf_node_idx >= self.leaf_node_holder:
                break
        
        if len(tmp_list) != 0:
            leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)
        
        return leaf_node_vec
    
    def get_ratio(self):
        """Get current space utilization ratio."""
        return self.space.get_ratio()
    
    def get_state_snapshot(self):
        """
        Get a lightweight state snapshot for MCTS simulation.
        Much faster than copy.deepcopy() - only copies essential data.
        
        Returns:
            dict: State snapshot that can be used with restore_state_snapshot()
        """
        return {
            'space_state': self.space.get_state(),
            'buffer': [{'size': item['size'].copy() if hasattr(item['size'], 'copy') else list(item['size']),
                       'density': item['density'],
                       'original': item['original']} for item in self.buffer],
            'item_queue': [{'size': item['size'].copy() if hasattr(item['size'], 'copy') else list(item['size']),
                           'density': item['density'],
                           'original': item['original']} for item in self.item_queue],
            'done': self.done,
            'packed': [p.copy() if hasattr(p, 'copy') else list(p) for p in self.packed] if hasattr(self, 'packed') else [],
        }
    
    def restore_state_snapshot(self, snapshot):
        """
        Restore environment state from a lightweight snapshot.
        
        Args:
            snapshot: State snapshot from get_state_snapshot()
        """
        self.space.restore_state(snapshot['space_state'])
        self.buffer = [{'size': list(item['size']),
                       'density': item['density'],
                       'original': item['original']} for item in snapshot['buffer']]
        self.item_queue = [{'size': list(item['size']),
                           'density': item['density'],
                           'original': item['original']} for item in snapshot['item_queue']]
        self.done = snapshot['done']
        self.packed = [list(p) for p in snapshot['packed']] if 'packed' in snapshot else []
    
    def lightweight_copy(self):
        """
        Create a lightweight copy of the environment for MCTS simulation.
        Uses state snapshot instead of deep copy for better performance.
        
        Returns:
            PackingDiscreteWithBuffer: A new environment instance with copied state
        """
        # Create new environment with same configuration
        new_env = PackingDiscreteWithBuffer(
            setting=self.setting,
            container_size=self.bin_size,
            item_set=self.item_set,
            internal_node_holder=self.internal_node_holder,
            leaf_node_holder=self.leaf_node_holder,
            next_holder=self.next_holder,
            buffer_size=self.buffer_size,
            total_items=self.total_items,
            LNES=self.LNES
        )
        # Restore state from snapshot
        new_env.restore_state_snapshot(self.get_state_snapshot())
        return new_env
    
    def copy(self):
        """Create a deep copy of the environment for simulation."""
        # Use lightweight copy for better performance
        return self.lightweight_copy()


# For compatibility with gym.make()
PackingDiscreteBuffer = PackingDiscreteWithBuffer
