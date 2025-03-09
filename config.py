import jax.numpy as jnp
import os

CONTEXT_LENGTH = 512
BATCH_SIZE = 512
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
GRADIENT_CLIP_NORM = 1.0
DTYPE = jnp.bfloat16
PARALLEL_PROCESSING = 8
TOKENIZED_DATASET_PATH = "/mnt/data/tokenized_dataset"
TOKENIZER_NAME = "gpt2"
CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), "checkpoints")
LOG_STEPS = 10
EVAL_STEPS = 5000
MESH_SHAPE = (16, 1)  # (data, expert)

VOCAB_SIZE = 50257
VOCAB_SIZE = ((VOCAB_SIZE + 127) // 128) * 128

DATASET_CONFIG = {
    'path': 'wikitext',#'HuggingFaceFW/fineweb',
    'name': 'wikitext-103-v1',#'sample-10BT',
    'split': 'train',
}

MODEL_CONFIG = {
    'd_model': 768,
    'hidden_dim': 4096,
    'num_layers': 16,
    'num_heads': 16,
    'head_dim': 64,
    'vocab_size': VOCAB_SIZE,
    'num_experts': 8,
    'num_shared_experts': 1,
    'top_k': 2,
    'capacity_factor': 2.0,
    'min_expert_capacity': 8,
    'max_group_size': 4096,
    'router_z_loss_coef': 1e-3,
    'router_balance_loss_coef': 1e-4,
    'dtype': DTYPE,
    'training': True,
    'use_gradient_checkpointing': True,
}

def calculate_model_params(config: dict) -> dict:
    """
    Calculate number of parameters for each component based on model.py implementation.
    Returns a dictionary with parameter counts for each component.
    """
    params = {}
    
    # Token embeddings (model.py: self.token_embedding)
    params['token_embedding'] = config['vocab_size'] * config['d_model']
    
    # Per block parameters
    block_params = {}
    
    # 1. Attention block
    # RMSNorm (model.py: self.attn_norm)
    block_params['attn_norm'] = config['d_model']
    
    # MultiHeadAttention (model.py: self.attention)
    # in_proj (3 matrices for Q,K,V)
    block_params['attention_in_proj'] = 3 * config['d_model'] * config['num_heads'] * config['head_dim']
    # out_proj
    block_params['attention_out_proj'] = config['num_heads'] * config['head_dim'] * config['d_model']
    
    # 2. MoE block
    # RMSNorm (model.py: self.moe_norm)
    block_params['moe_norm'] = config['d_model']
    
    # Router (model.py: Router class)
    block_params['router_weight'] = config['d_model'] * config['num_experts']
    block_params['router_bias'] = config['num_experts']
    
    # FeedForward experts (model.py: FeedForward class)
    # Regular experts
    expert_params = (
        # keys matrix
        config['d_model'] * config['hidden_dim'] +
        # values matrix
        config['hidden_dim'] * config['d_model']
    )
    block_params['ff_experts'] = expert_params * config['num_experts']
    
    # Shared experts
    if config['num_shared_experts'] > 0:
        block_params['shared_experts'] = expert_params * config['num_shared_experts']
    else:
        block_params['shared_experts'] = 0
    
    # Total params per block
    params['per_block'] = sum(block_params.values())
    
    # Final layer norm (model.py: self.final_norm)
    params['final_norm'] = config['d_model']
    
    # Output projection (model.py: self.lm_head)
    params['lm_head'] = config['d_model'] * config['vocab_size']
    
    # Calculate totals
    params['total_block_params'] = params['per_block'] * config['num_layers']
    params['total'] = (
        params['token_embedding'] +
        params['total_block_params'] +
        params['final_norm'] +
        params['lm_head']
    )
    
    # Add block breakdown for reference
    params['block_breakdown'] = block_params
    
    return params

def format_params(n: int) -> str:
    """Format parameter count in human readable form."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)

if __name__ == "__main__":
    params = calculate_model_params(MODEL_CONFIG)
    
    print("\nModel Parameter Breakdown:")
    print("-" * 50)
    print(f"Token Embedding: {format_params(params['token_embedding'])}")
    print("\nPer Block:")
    for name, count in params['block_breakdown'].items():
        print(f"  {name}: {format_params(count)}")
    print(f"\nParameters per block: {format_params(params['per_block'])}")
    print(f"Total block parameters ({MODEL_CONFIG['num_layers']} blocks): {format_params(params['total_block_params'])}")
    print(f"Final norm: {format_params(params['final_norm'])}")
    print(f"LM head: {format_params(params['lm_head'])}")
    print("-" * 50)
    print(f"Total Parameters: {format_params(params['total'])}")
    
    # Calculate effective parameters per token (considering top_k)
    total_experts = MODEL_CONFIG['num_experts'] + MODEL_CONFIG['num_shared_experts']
    experts_per_token = MODEL_CONFIG['top_k'] + MODEL_CONFIG['num_shared_experts']
    expert_ratio = experts_per_token / total_experts
    effective_params = (
        params['token_embedding'] +
        (params['total_block_params'] * expert_ratio) +
        params['final_norm'] +
        params['lm_head']
    )
    print(f"Effective parameters per token: {format_params(effective_params)}")
    
    print("\nVRAM Usage Estimation:")
    print("-" * 50)
    
    # Get basic info
    batch_size = BATCH_SIZE
    seq_len = CONTEXT_LENGTH
    d_model = MODEL_CONFIG['d_model']
    num_layers = MODEL_CONFIG['num_layers']
    hidden_dim = MODEL_CONFIG['hidden_dim']
    num_experts = MODEL_CONFIG['num_experts']
    num_shared = MODEL_CONFIG['num_shared_experts']
    dtype_size = 2  # bfloat16 is 2 bytes
    
    # 1. Model Parameters VRAM
    param_bytes = params['total'] * dtype_size
    
    # 2. Optimizer States (Adam/AdamW has 2 states per param + master copy in fp32)
    optimizer_bytes = param_bytes * 4  # 2 states in fp32 (8 bytes) + 1 master copy in fp32
    
    # 3. Activation Memory
    # Key activations per layer:
    activations_per_layer = {
        'attention_qkv': 3 * batch_size * seq_len * d_model,  # Q, K, V
        'attention_scores': batch_size * MODEL_CONFIG['num_heads'] * seq_len * seq_len,  # Attention scores
        'moe_router_logits': batch_size * seq_len * num_experts,  # Router logits
        'moe_dispatch': batch_size * seq_len * num_experts * (MODEL_CONFIG['top_k'] + 1),  # Dispatch tensors
        'moe_expert_inputs': batch_size * seq_len * hidden_dim * (MODEL_CONFIG['top_k'] + num_shared),  # Expert inputs
        'residuals': batch_size * seq_len * d_model,  # Layer residuals
    }
    
    total_activations = sum(activations_per_layer.values()) * num_layers
    activation_bytes = total_activations * dtype_size
    
    # 4. Gradient Memory (roughly same as activations during backward pass)
    gradient_bytes = activation_bytes
    
    # 5. Additional memory for gradient checkpointing (if enabled)
    if MODEL_CONFIG['use_gradient_checkpointing']:
        checkpoint_bytes = activation_bytes * 0.2  # Rough estimate: 20% of activation memory
    else:
        checkpoint_bytes = 0
    
    # Total VRAM
    total_bytes = param_bytes + optimizer_bytes + activation_bytes + gradient_bytes + checkpoint_bytes
    
    def format_bytes(bytes):
        if bytes >= 1e9:
            return f"{bytes/1e9:.2f} GB"
        elif bytes >= 1e6:
            return f"{bytes/1e6:.2f} MB"
        elif bytes >= 1e3:
            return f"{bytes/1e3:.2f} KB"
        return f"{bytes} B"
    
    print(f"Model Parameters: {format_bytes(param_bytes)}")
    print(f"Optimizer States: {format_bytes(optimizer_bytes)}")
    print(f"Activation Memory: {format_bytes(activation_bytes)}")
    print(f"Gradient Memory: {format_bytes(gradient_bytes)}")
    if MODEL_CONFIG['use_gradient_checkpointing']:
        print(f"Gradient Checkpointing Memory: {format_bytes(checkpoint_bytes)}")
    print("-" * 50)
    print(f"Total VRAM Usage: {format_bytes(total_bytes)}")
    print(f"VRAM per TPU/GPU: {format_bytes(total_bytes / (MESH_SHAPE[0] * MESH_SHAPE[1]))}")
    
    # Memory efficiency metrics
    print("\nMemory Efficiency Metrics:")
    print("-" * 50)
    tokens_per_batch = batch_size * seq_len
    print(f"Tokens per batch: {tokens_per_batch}")
    print(f"VRAM per token: {format_bytes(total_bytes / tokens_per_batch)}")
    print(f"Parameters per VRAM GB: {format_params(params['total'] / (total_bytes / 1e9))}/GB")