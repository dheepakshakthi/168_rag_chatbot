"""
SmolLM2:1.7B Model Implementation in PyTorch
A standalone implementation of the SmolLM2 architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return self.weight * x


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = config['max_position_embeddings']
        
        assert self.hidden_size % self.num_heads == 0
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, self.max_seq_len)
        self.attention_dropout = nn.Dropout(config.get('attention_dropout', 0.0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rotary_emb(q, k, seq_len)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Compute output
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.o_proj(context)
        return output


class MLP(nn.Module):
    """Feed-Forward Network with SwiGLU activation"""
    def __init__(self, config):
        super().__init__()
        hidden_size = config['hidden_size']
        intermediate_size = config['intermediate_size']
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(config.get('hidden_dropout', 0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: gate * SiLU(up)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        return output


class TransformerBlock(nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config['hidden_size'])
        self.post_attention_layernorm = RMSNorm(config['hidden_size'])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SmolLM2Model(nn.Module):
    """SmolLM2:1.7B Language Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['num_hidden_layers'])
        ])
        
        # Final layer norm
        self.norm = RMSNorm(self.hidden_size)
        
        # LM head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Tie weights (embedding and output projection share weights)
        self.lm_head.weight = self.embed_tokens.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
        
        # Convert to attention mask format (additive mask)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        attention_mask = attention_mask & ~causal_mask
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate text using the model"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids


def create_smollm2_1_7b():
    """Create SmolLM2:1.7B model with default configuration"""
    config = {
        'vocab_size': 49152,
        'hidden_size': 2048,
        'intermediate_size': 8192,
        'num_hidden_layers': 24,
        'num_attention_heads': 32,
        'max_position_embeddings': 2048,
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
    }
    
    model = SmolLM2Model(config)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"SmolLM2:1.7B Model Created")
    print(f"Total Parameters: {num_params:,} ({num_params / 1e9:.2f}B)")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Num Layers: {config['num_hidden_layers']}")
    print(f"Num Heads: {config['num_attention_heads']}")
    print(f"Vocab Size: {config['vocab_size']}")
    
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_smollm2_1_7b()
    
    # Example forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 49152, (batch_size, seq_len))
    
    print("\nRunning example forward pass...")
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {model.vocab_size}]")
    
    # Example generation
    print("\nExample generation (with random weights):")
    prompt = torch.randint(0, 49152, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Generated sequence length: {generated.shape[1]} tokens")
