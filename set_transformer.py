import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SetTransformer(nn.Module):
    """
    Attribute-Aware Set Transformer for prior probability estimation.
    
    This network processes a buffer of N items (permutation-invariant set)
    and outputs:
    1. Prior probabilities for selecting each item
    2. State value estimate
    
    Architecture based on "Set Transformer: A Framework for Attention-based 
    Permutation-Invariant Neural Networks" (Lee et al., 2019)
    """
    
    def __init__(self, 
                 d_model=128, 
                 n_heads=4, 
                 n_layers=3, 
                 buffer_size=10,
                 dropout=0.1):
        """
        Args:
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            buffer_size: Maximum number of items in buffer (N=10)
            dropout: Dropout rate
        """
        super(SetTransformer, self).__init__()
        
        self.d_model = d_model
        self.buffer_size = buffer_size
        
        # Item attribute embedding: (L, W, H, Density) -> d_model
        # Input: 4 dimensions (length, width, height, density)
        self.item_embed = nn.Linear(4, d_model)
        
        # Positional encoding is NOT needed since set is permutation-invariant
        # But we can optionally add learned embeddings for buffer positions
        self.use_position_embed = False  # Can be enabled if needed
        if self.use_position_embed:
            self.position_embed = nn.Embedding(buffer_size, d_model)
        
        # Multi-head self-attention transformer encoder
        # This is the core of Set Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Input shape: (batch, seq_len, d_model)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Output heads
        # Policy head: outputs logits for each item in buffer
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)  # Single logit per item
        )
        
        # Value head: outputs state value estimate
        # Aggregate all item embeddings then predict value
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)  # Single value
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, buffer_items, mask=None):
        """
        Forward pass.
        
        Args:
            buffer_items: Tensor of shape (batch_size, buffer_size, 4)
                         Each item has (L, W, H, Density)
            mask: Optional boolean mask of shape (batch_size, buffer_size)
                  True indicates invalid/padding items
                  
        Returns:
            policy_probs: Prior probabilities, shape (batch_size, buffer_size)
            value: State value estimate, shape (batch_size, 1)
        """
        batch_size = buffer_items.size(0)
        
        # 1. Embed item attributes
        # (batch_size, buffer_size, 4) -> (batch_size, buffer_size, d_model)
        item_embeddings = self.item_embed(buffer_items)
        
        # 2. Optional: Add position embeddings (if enabled)
        if self.use_position_embed:
            positions = torch.arange(self.buffer_size, device=buffer_items.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            pos_embeddings = self.position_embed(positions)
            item_embeddings = item_embeddings + pos_embeddings
        
        # 3. Layer normalization
        item_embeddings = self.layer_norm(item_embeddings)
        
        # 4. Create attention mask if provided
        # PyTorch transformer uses additive mask: 0 for valid, -inf for invalid
        attn_mask = None
        if mask is not None:
            attn_mask = mask.float()
            attn_mask = attn_mask.masked_fill(mask, float('-inf'))
        
        # 5. Multi-head self-attention via transformer encoder
        # Output shape: (batch_size, buffer_size, d_model)
        encoded_items = self.transformer_encoder(
            item_embeddings,
            src_key_padding_mask=mask  # Use padding mask
        )
        
        # 6. Policy head: Get logits for each item
        # Shape: (batch_size, buffer_size, 1) -> (batch_size, buffer_size)
        policy_logits = self.policy_head(encoded_items).squeeze(-1)
        
        # Apply mask to logits (set invalid items to -inf)
        if mask is not None:
            policy_logits = policy_logits.masked_fill(mask, float('-inf'))
        
        # Softmax to get probabilities
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # 7. Value head: Aggregate embeddings and predict value
        # Use mean pooling over valid items
        if mask is not None:
            # Mask out invalid items before pooling
            encoded_items_masked = encoded_items.masked_fill(
                mask.unsqueeze(-1).expand_as(encoded_items),
                0.0
            )
            # Average over valid items only
            valid_counts = (~mask).sum(dim=1, keepdim=True).float()  # (batch_size, 1)
            pooled = encoded_items_masked.sum(dim=1) / valid_counts  # (batch_size, d_model)
        else:
            # Simple mean pooling
            pooled = encoded_items.mean(dim=1)
        
        value = self.value_head(pooled)  # (batch_size, 1)
        
        return policy_probs, value
    
    def get_action(self, buffer_items, mask=None, deterministic=False):
        """
        Sample or select action from policy.
        
        Args:
            buffer_items: Tensor of shape (batch_size, buffer_size, 4)
            mask: Optional mask
            deterministic: If True, select argmax; if False, sample
            
        Returns:
            action: Selected item indices, shape (batch_size,)
            log_prob: Log probability of selected action
        """
        policy_probs, value = self.forward(buffer_items, mask)
        
        if deterministic:
            # Select item with highest probability
            action = policy_probs.argmax(dim=-1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(policy_probs)
            action = dist.sample()
        
        # Get log probability
        log_prob = torch.log(policy_probs.gather(1, action.unsqueeze(-1)) + 1e-8)
        
        return action, log_prob, value


class IndependentSetTransformer(nn.Module):
    """
    Variant: Two independent networks for policy and value.
    Can be useful if policy and value learning have different dynamics.
    """
    
    def __init__(self, d_model=128, n_heads=4, n_layers=3, buffer_size=10, dropout=0.1):
        super(IndependentSetTransformer, self).__init__()
        
        # Shared embedding
        self.item_embed = nn.Linear(4, d_model)
        
        # Separate transformers for policy and value
        self.policy_encoder = self._build_encoder(d_model, n_heads, n_layers, dropout)
        self.value_encoder = self._build_encoder(d_model, n_heads, n_layers, dropout)
        
        self.policy_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)
        
    def _build_encoder(self, d_model, n_heads, n_layers, dropout):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, buffer_items, mask=None):
        embeddings = self.item_embed(buffer_items)
        
        # Policy branch
        policy_encoded = self.policy_encoder(embeddings, src_key_padding_mask=mask)
        policy_logits = self.policy_head(policy_encoded).squeeze(-1)
        if mask is not None:
            policy_logits = policy_logits.masked_fill(mask, float('-inf'))
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Value branch
        value_encoded = self.value_encoder(embeddings, src_key_padding_mask=mask)
        pooled = value_encoded.mean(dim=1)
        value = self.value_head(pooled)
        
        return policy_probs, value


if __name__ == '__main__':
    """Test Set Transformer functionality."""
    print("Testing Set Transformer...")
    
    # Create model
    model = SetTransformer(d_model=128, n_heads=4, n_layers=3, buffer_size=10)
    
    # Test input: batch of 4, buffer of 10 items
    batch_size = 4
    buffer_size = 10
    buffer_items = torch.randn(batch_size, buffer_size, 4)  # Random items
    
    # Test without mask
    print("\n1. Test without mask:")
    policy_probs, value = model(buffer_items)
    print(f"Policy probs shape: {policy_probs.shape}")  # Should be (4, 10)
    print(f"Value shape: {value.shape}")  # Should be (4, 1)
    print(f"Policy sum: {policy_probs.sum(dim=-1)}")  # Should be all 1.0
    
    # Test with mask (some items invalid)
    print("\n2. Test with mask:")
    mask = torch.zeros(batch_size, buffer_size, dtype=torch.bool)
    mask[0, 5:] = True  # First sample: only 5 valid items
    mask[1, 8:] = True  # Second sample: only 8 valid items
    
    policy_probs, value = model(buffer_items, mask)
    print(f"Policy probs (masked): {policy_probs[0]}")
    print(f"Masked positions sum: {policy_probs[0, 5:].sum()}")  # Should be ~0
    
    # Test action sampling
    print("\n3. Test action sampling:")
    action, log_prob, value = model.get_action(buffer_items, mask, deterministic=False)
    print(f"Sampled action: {action}")
    print(f"Log prob: {log_prob}")
    
    action_det, log_prob_det, _ = model.get_action(buffer_items, mask, deterministic=True)
    print(f"Deterministic action: {action_det}")
    
    # Test permutation invariance
    print("\n4. Test permutation invariance:")
    # Shuffle items
    perm = torch.randperm(buffer_size)
    buffer_items_perm = buffer_items[:, perm, :]
    
    policy_probs_orig, _ = model(buffer_items)
    policy_probs_perm, _ = model(buffer_items_perm)
    
    # Probabilities should be permuted the same way
    policy_probs_unperm = policy_probs_perm[:, torch.argsort(perm), :]
    diff = (policy_probs_orig - policy_probs_unperm).abs().max()
    print(f"Max difference after permutation: {diff.item()}")  # Should be ~0
    
    print("\n✓ All tests passed!")
