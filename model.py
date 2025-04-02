import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

@partial(nnx.vmap, in_axes=(0, None, 0, None))
def _apply_mask(score_matrix: jnp.ndarray, seq_len: int, attn_mask: jnp.ndarray | None = None, is_causal: bool = True) -> jnp.ndarray:
    # Pre-compute causal mask
    if is_causal:
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=score_matrix.dtype))
    else:
        causal_mask = jnp.ones((seq_len, seq_len), dtype=score_matrix.dtype)

    # Combine masks using element-wise multiplication
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(1, seq_len).astype(score_matrix.dtype)
        combined_mask = causal_mask * attn_mask
    else:
        combined_mask = causal_mask

    # Add self-attention by setting diagonal to 1
    combined_mask = combined_mask.at[jnp.diag_indices(seq_len)].set(1.0)

    # Apply continuous masking using linear interpolation
    mask_scale = -(1 - combined_mask) * 1e9  # 0 → 0 penalty, 1 → -1e9 penalty
    return score_matrix + mask_scale

def orthogonal_init(scale=1.0):
    """Create an orthogonal initializer with the given scale."""
    def init(key, shape, dtype=jnp.float32):
        if len(shape) < 2:
            raise ValueError("Orthogonal initialization requires at least a 2D shape")
        
        # Get the shape for the matrix
        rows, cols = shape[-2:]
        
        # Generate a random matrix
        key, subkey = jax.random.split(key)
        unstructured = jax.random.normal(subkey, shape, dtype)
        
        # Compute the QR decomposition
        q, r = jnp.linalg.qr(unstructured)
        
        # Make Q uniform
        q = q * jnp.sign(jnp.diag(r))
        
        # Ensure q has the right size
        if rows < cols:
            q = q.T
        
        # Apply scaling
        q = scale * q
        
        return q.astype(dtype)
    return init

class DyT(nnx.Module):
    def __init__(self, 
        dim: int,
        init_alpha: float = 1.0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.dim = dim
        self.dtype = dtype
        
        key = rngs.params()
        
        # Initialize alpha as a learnable scalar
        self.alpha = nnx.Param(
            jnp.ones((1,), dtype=dtype) * init_alpha,
            sharding=(None,),
            dtype=self.dtype,
        )
        
        # Initialize gamma (scale) and beta (bias) parameters
        self.gamma = nnx.Param(
            jnp.ones((dim,), dtype=dtype),
            sharding=(None,),
            dtype=self.dtype,
        )
        self.beta = nnx.Param(
            jnp.zeros((dim,), dtype=dtype),
            sharding=(None,),
            dtype=self.dtype,
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Apply dynamic tanh normalization
        x = jnp.tanh(self.alpha.value * x)
        
        # Apply affine transformation
        return self.gamma.value * x + self.beta.value

class RotaryEmbedding(nnx.Module):
    def __init__(self, 
        dim: int, 
        base: int = 10000,
        dtype: jnp.dtype = jnp.bfloat16,
        training: bool = False,
    ):
        self.dim = dim
        self.dtype = dtype
        self.base = base
        self.training = training
    
    def get_rotary_cache(self, seq_len: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        inv_freq = 1.0 / (self.base ** (2 * jnp.arange(0, self.dim // 2) / self.dim))
        positions = jnp.arange(seq_len, dtype=self.dtype)
        angles = positions[:, None] * inv_freq[None, :]
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)
        return (
            cos.reshape(1, seq_len, 1, cos.shape[-1]),
            sin.reshape(1, seq_len, 1, sin.shape[-1])
        )

    def __call__(self, x: jnp.ndarray, seq_len: int | None = None) -> jnp.ndarray:
        seq_len = seq_len if seq_len is not None else x.shape[1]
        cos, sin = self.get_rotary_cache(seq_len)
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)

        x_out = jnp.concatenate([
            x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin,
            x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos
        ], axis=-1)

        return x_out.reshape(x.shape)

    def rotate_queries_and_keys(self, q: jnp.ndarray, k: jnp.ndarray, seq_len: int | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        seq_len = seq_len if seq_len is not None else q.shape[1]
        return self.__call__(q, seq_len), self.__call__(k, seq_len)

class MultiHeadAttention(nnx.Module):
    def __init__(self, 
        num_heads: int, 
        d_model: int,
        head_dim: int,
        dtype: jnp.dtype = jnp.bfloat16,
        training: bool = False,
        need_attention_mask: bool = True,
        is_causal: bool = True,
        init_fn = None,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = head_dim
        self.dtype = dtype
        self.training = training
        self.need_attention_mask = need_attention_mask
        self.is_causal = is_causal
        
        key = rngs.params()
        
        if init_fn is None:
            init_fn = nnx.initializers.normal(stddev=0.02, dtype=self.dtype)

        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            dtype=self.dtype,
            training=self.training,
        )

        self.in_proj = nnx.Param(
            init_fn(key, (3, self.d_model, self.num_heads, self.head_dim)),
            sharding=(None, None, None, None),
            dtype=self.dtype,
        )

        self.out_proj = nnx.Param(
            init_fn(key, (self.num_heads, self.head_dim, self.d_model)),
            sharding=(None, None, None),
            dtype=self.dtype,
        )
    
    def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape

        qkv = jnp.einsum('bsd,tdhm->tbshm', x, self.in_proj.value) # reduce
        q, k, v = qkv

        q, k = self.rotary.rotate_queries_and_keys(q, k)

        q = jnp.einsum('bshd->bhsd', q)
        k = jnp.einsum('bshd->bhsd', k)
        v = jnp.einsum('bshd->bhsd', v)

        scores = jnp.einsum('bnqd,bnkd->bnqk', q, k) / jnp.sqrt(self.head_dim)
        
        # Handle both causal and attention masking in one step
        if self.need_attention_mask:
            if attn_mask is not None and attn_mask.ndim == 2 and attn_mask.shape[0] == batch_size:
                if attn_mask.shape[1] > seq_len:
                    attn_mask = attn_mask[:, :seq_len]
            scores = _apply_mask(scores, seq_len, attn_mask, self.is_causal)

        attn_weights = nnx.softmax(scores, axis=-1)

        output = jnp.einsum('bnqk,bnkd,ndo->bqo', attn_weights, v, self.out_proj.value)

        return output

class FeedForward(nnx.Module):
    def __init__(self, 
        d_model: int,
        hidden_dim: int,
        num_experts: int = 1,
        dtype: jnp.dtype = jnp.bfloat16,
        training: bool = False,
        init_fn = None,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.dtype = dtype
        self.training = training

        key = rngs.params()
        
        if init_fn is None:
            init_fn = nnx.initializers.normal(stddev=0.02, dtype=self.dtype)
        
        self.keys = nnx.Param(
            init_fn(key, (self.num_experts, self.d_model, self.hidden_dim)),
            sharding=('expert', None, None),
            dtype=self.dtype,
        )
        self.values = nnx.Param(
            init_fn(key, (self.num_experts, self.hidden_dim, self.d_model)),
            sharding=('expert', None, None),
            dtype=self.dtype,
        )

        self.act = nnx.gelu
    
    def process_by_indices(self, x: jnp.ndarray, indices: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        selected_keys = self.keys.value[indices]
        selected_values = self.values.value[indices]

        # [batch, seq, d_model] @ [batch, seq, top_k, d_model, hidden_dim] -> [batch, seq, top_k, hidden_dim]
        hidden = jnp.einsum('bsd,bskdh->bskh', x, selected_keys)
        
        # Add bias and apply activation
        hidden = self.act(hidden)
        
        # [batch, seq, top_k, hidden_dim] @ [batch, seq, top_k, d_model] -> [batch, seq, d_model]
        output = jnp.einsum('bskh,bskhd,bsk->bsd', hidden, selected_values, weights)
        
        return output

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Add sharding hints for expert-parallel computation
        x = jax.lax.with_sharding_constraint(
            x, jax.sharding.PartitionSpec('data', 'expert', None, None)
        )

        hidden = jax.lax.with_sharding_constraint(
            jnp.einsum('besd,edh->besh', x, self.keys.value),
            jax.sharding.PartitionSpec('data', 'expert', None, None)
        )

        hidden = self.act(hidden)
        
        output = jax.lax.with_sharding_constraint(
            jnp.einsum('besh,ehd->besd', hidden, self.values.value),
            jax.sharding.PartitionSpec('data', 'expert', None, None)
        )
        
        return output

class Router(nnx.Module):
    def __init__(self, 
        d_model: int,
        num_experts: int,
        z_loss_coef: float = 1e-3,
        balance_loss_coef: float = 1e-4,
        top_k: int = 2,
        dtype: jnp.dtype = jnp.bfloat16,
        training: bool = False,
        init_fn = None,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.d_model = d_model
        self.num_experts = num_experts
        self.z_loss_coef = z_loss_coef
        self.balance_loss_coef = balance_loss_coef
        self.top_k = top_k
        self.dtype = dtype
        self.training = training

        key = rngs.params()

        if init_fn is None:
            init_fn = orthogonal_init(scale=1.0)

        # Replace direct parameter with linear layer including bias
        self.gate_weight = nnx.Param(
            init_fn(key, (self.d_model, self.num_experts)),
            sharding=(None, None),
            dtype=self.dtype,
        )
        self.gate_bias = nnx.Param(
            jnp.zeros((1, 1, self.num_experts), dtype=self.dtype),
            sharding=(None, None, None),
            dtype=self.dtype,
        )

    def __call__(self, x: jnp.ndarray, expert_capacity: int | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        # x shape: (groups, size, d_model)
        x = jax.lax.with_sharding_constraint(
            x, jax.sharding.PartitionSpec('data', None, None)
        )
        gating_logits = jax.lax.with_sharding_constraint(
            jnp.einsum('gsd,de->gse', x, self.gate_weight.value),
            jax.sharding.PartitionSpec('data', None, None)
        )
        gating_logits = gating_logits + self.gate_bias
        
        gating_probs = nnx.softmax(gating_logits)
        if self.training:
            @partial(jax.vmap, in_axes=(0, None))
            @partial(jax.vmap, in_axes=(0, None))
            def vmapped_approx_max_k(probs, k):
                return jax.lax.approx_max_k(probs, k)
            expert_gate, expert_index = vmapped_approx_max_k(gating_probs, self.top_k)
        else:
            expert_gate, expert_index = jax.lax.top_k(gating_probs, self.top_k)
        
        if expert_capacity is None:
            return expert_gate, expert_index, None
        
        expert_mask = nnx.one_hot(expert_index, num_classes=self.num_experts)
        combined_expert_mask = jnp.sum(expert_mask, axis=2)
        
        loss = None
        if self.training:
            z_loss_per_batch = jnp.mean(nnx.logsumexp(gating_logits, axis=-1) ** 2, axis=1)  # shape: (batch,)
            
            probs_mean_per_batch = jnp.mean(gating_probs, axis=1)  # shape: (batch, num_experts)
            mask_mean_per_batch = jnp.mean(combined_expert_mask, axis=1)  # shape: (batch, num_experts)
            balance_loss_per_batch = jnp.mean(
                probs_mean_per_batch * mask_mean_per_batch * (mask_mean_per_batch.shape[-1] ** 2),
                axis=-1  # Reduce over experts dimension to get shape (batch,)
            )
            
            loss = balance_loss_per_batch * self.balance_loss_coef + z_loss_per_batch * self.z_loss_coef
        
        position_in_expert = jnp.cumsum(expert_mask, axis=1) * expert_mask
        valid_assignment = jnp.less(position_in_expert, expert_capacity)
        
        # Apply valid assignments to expert gates
        expert_gate_valid = expert_gate[..., None] * valid_assignment.astype(expert_gate.dtype)
        
        # Create combine tensor with position-wise assignments
        combine_tensor_per_assignment = (
            expert_gate_valid[..., None] *
            nnx.one_hot(position_in_expert, num_classes=expert_capacity)
        )
        
        # Sum across top_k dimension to get final combine tensor
        # shape: (groups, size, experts, capacity)
        combine_tensor_per_assignment = jax.lax.with_sharding_constraint(
            combine_tensor_per_assignment,
            jax.sharding.PartitionSpec('data', None, None, None)
        )
        # Replace slow reduction with fast einsum
        combine_tensor = jnp.einsum('gstec->gsec', combine_tensor_per_assignment)
        
        # Remove the first position (zero position) from capacity dimension
        combine_tensor = combine_tensor[..., 1:]
        
        # Create dispatch mask from combine tensor
        dispatch_mask = combine_tensor.astype(bool)

        return combine_tensor, dispatch_mask, loss

class MixtureLayer(nnx.Module):
    def __init__(self, 
        d_model: int,
        hidden_dim: int,
        num_total_experts: int,
        num_shared_experts: int,
        top_k: int = 2,
        capacity_factor: float = 2.0,
        min_expert_capacity: int = 8,
        max_group_size: int = 4096,
        router_z_loss_coef: float = 1e-3,
        router_balance_loss_coef: float = 1e-4,
        dtype: jnp.dtype = jnp.bfloat16,
        training: bool = False,
        init_fn = None,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_total_experts = num_total_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.min_expert_capacity = min_expert_capacity
        self.max_group_size = max_group_size
        self.router_z_loss_coef = router_z_loss_coef
        self.router_balance_loss_coef = router_balance_loss_coef
        self.dtype = dtype
        self.training = training

        key = rngs.params()

        if init_fn is None:
            init_fn = nnx.initializers.normal(stddev=0.02, dtype=self.dtype)

        self.num_ff_experts = self.num_total_experts
        assert self.num_ff_experts >= 0, "Total special experts exceeds num_experts"

        self.router = Router(
            d_model=self.d_model,
            num_experts=self.num_total_experts,
            z_loss_coef=self.router_z_loss_coef,
            balance_loss_coef=self.router_balance_loss_coef,
            top_k=self.top_k,
            dtype=self.dtype,
            training=self.training,
            rngs=rngs
        )
        
        if self.num_ff_experts > 0:
            self.feedforward_experts = FeedForward(
                d_model=self.d_model,
                hidden_dim=self.hidden_dim,
                num_experts=self.num_ff_experts,
                dtype=self.dtype,
                training=self.training,
                init_fn=init_fn,
                rngs=rngs
            )
        
        if self.num_shared_experts > 0:
            self.shared_experts = FeedForward(
                d_model=self.d_model,
                hidden_dim=self.hidden_dim,
                num_experts=self.num_shared_experts,
                dtype=self.dtype,
                training=self.training,
                init_fn=init_fn,
                rngs=rngs
            )
    
    def _compute_group_size(self, batch_size, seq_len):
        if self.training:
            group_size = seq_len
            num_groups = batch_size
            expert_capacity = int((group_size * self.top_k * self.capacity_factor) / self.num_total_experts)
            return group_size, num_groups, expert_capacity

        batch_size_int = int(batch_size)
        seq_len_int = int(seq_len)
        num_tokens = batch_size_int * seq_len_int
        sqrt_tokens = int(num_tokens ** 0.5)
        group_size = min(self.max_group_size, max(32, sqrt_tokens))
        num_groups = (num_tokens + group_size - 1) // group_size

        expert_capacity = int((group_size * self.top_k * self.capacity_factor) / self.num_total_experts)
        
        min_capacity = max(
            self.min_expert_capacity,
            self.top_k,
            int(group_size * 0.01)
        )
        
        max_capacity = min(
            group_size,               # At most all tokens in a group
            int(group_size * 0.25)    # Cap at 25% of group size to prevent excessive memory usage
        )
        
        # Final expert capacity within bounds
        expert_capacity = min(max_capacity, max(expert_capacity, min_capacity))
        
        return group_size, num_groups, expert_capacity
    
    def _process_shared_experts(self, x: jnp.ndarray) -> jnp.ndarray:
        """Process shared experts in parallel."""
        if not self.num_shared_experts:
            return jnp.zeros_like(x)
        
        x_expanded = x[:, None, :, :]
        shared_outputs = self.shared_experts(x_expanded)
        output = jnp.mean(shared_outputs, axis=1)

        return output
    
    def _group_inputs(self, x: jnp.ndarray) -> tuple[jnp.ndarray, int, tuple[int, int], bool]:
        # Extract static dimensions where possible to help JAX optimization
        original_batch_size, original_seq_len, d_model = x.shape
        
        # Calculate group parameters - keep these calculations static where possible
        # to enable better XLA fusion
        group_size, num_groups, expert_capacity = self._compute_group_size(original_batch_size, original_seq_len)

        if self.training:
            return x, expert_capacity, (original_batch_size, original_seq_len), False
        
        # Save original shape for depadding
        original_shape = (original_batch_size, original_seq_len)
        
        # Calculate total tokens and required padding
        total_tokens = original_batch_size * original_seq_len
        padded_size = num_groups * group_size
        padding_needed = padded_size - total_tokens
        
        # Flag to indicate if padding was applied
        was_padded = padding_needed > 0
        
        # Fuse reshape and padding operations to minimize data movement
        if was_padded:
            # Use a single reshape-and-pad operation where possible
            # Reshape first then pad to make operation more efficient
            x_flat = jnp.reshape(x, (-1, d_model))
            
            # Add padding as a single operation
            x_padded = jnp.pad(
                x_flat,
                pad_width=((0, padding_needed), (0, 0)),
                mode='constant',
                constant_values=0
            )
            
            # Final reshape into groups
            x_grouped = jnp.reshape(x_padded, (num_groups, group_size, d_model))
        else:
            # When no padding needed, do direct reshape to minimize operations
            x_grouped = jnp.reshape(x, (num_groups, group_size, d_model))
        
        return x_grouped, expert_capacity, original_shape, was_padded

    def _degroup_outputs(self, x: jnp.ndarray, original_shape: tuple[int, int], was_padded: bool = True) -> jnp.ndarray:
        if self.training:
            return x

        original_batch_size, original_seq_len = original_shape

        # If no padding was applied, directly reshape back to original shape
        if not was_padded:
            return jnp.reshape(x, (original_batch_size, original_seq_len, x.shape[-1]))
        
        # Otherwise, we need to handle the padding removal
        # Calculate total tokens for efficient slicing
        total_tokens = original_batch_size * original_seq_len
        
        # Combine reshape and slice operations to minimize communication
        # First flatten the grouped tensor
        x_flat = jnp.reshape(x, (-1, x.shape[-1]))
        
        # Then slice to remove padding in one operation
        x_unpadded = x_flat[:total_tokens]
        
        # Final reshape to original dimensions
        # Use a static shape when possible to help JAX optimization
        output = jnp.reshape(x_unpadded, (original_batch_size, original_seq_len, x.shape[-1]))
        
        return output
    
    def _train_routing(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        x_grouped, expert_capacity, original_shape, was_padded = self._group_inputs(x)
        shared_output = self._process_shared_experts(x_grouped)
        
        combine_tensor, dispatch_mask, router_loss = self.router(x_grouped, expert_capacity)
        output = shared_output

        if self.num_ff_experts > 0:
            ff_dispatch = dispatch_mask[:,:,:self.num_ff_experts]
            ff_combine = combine_tensor[:,:,:self.num_ff_experts]
            
            # 1. DISPATCH: First all-to-all - tokens to experts
            # Shape annotations for clarity
            # ff_dispatch: [groups, size, experts, capacity]
            # x_grouped: [groups, size, d_model]
            dispatch_to_expert = jax.lax.with_sharding_constraint(
                jnp.einsum('gsec,gsd->gesd', ff_dispatch, x_grouped),
                jax.sharding.PartitionSpec('data', 'expert', None, None)
            )
            
            # 2. Process with experts (already sharded on expert dimension)
            expert_outputs = self.feedforward_experts(dispatch_to_expert)  # [experts, groups, size, d_model]
            
            # 3. COMBINE: Second all-to-all - results back to tokens
            # ff_combine: [groups, size, experts, capacity]
            # expert_outputs: [experts, groups, size, d_model]
            combine_from_expert = jax.lax.with_sharding_constraint(
                jnp.einsum('gesd,gsec->gsd', expert_outputs, ff_combine),
                jax.sharding.PartitionSpec('data', None, None)
            )

            output = output + combine_from_expert
        
        output = self._degroup_outputs(output, original_shape, was_padded)
        return output, router_loss
    
    def _eval_routing(self, x):
        output = self._process_shared_experts(x)
        
        expert_gate, expert_index, _ = self.router(x)
                
        if self.num_ff_experts > 0:
            ff_mask = expert_index < self.num_ff_experts
            
            ff_weights = expert_gate * ff_mask
            
            ff_weights_sum = jnp.sum(ff_weights, axis=-1, keepdims=True)
            ff_weights = ff_weights / (ff_weights_sum + 1e-9)
            
            ff_indices = jnp.where(ff_mask, expert_index, 0)
            
            ff_output = self.feedforward_experts.process_by_indices(
                x, ff_indices, ff_weights
            )
            
            output = output + ff_output

        return output
    
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        if self.training or x.shape[0] * x.shape[1] > 18:
            return self._train_routing(x)
        else:
            return self._eval_routing(x), None

class Block(nnx.Module):
    def __init__(self, 
        d_model: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2,
        capacity_factor: float = 2.0,
        min_expert_capacity: int = 8,
        max_group_size: int = 4096,
        router_z_loss_coef: float = 1e-3,
        router_balance_loss_coef: float = 1e-4,
        dtype: jnp.dtype = jnp.bfloat16,
        training: bool = False,
        use_gradient_checkpointing: bool = False,
        need_attention_mask: bool = True,
        is_causal: bool = True,
        init_fn = None,
        layer_idx: int = 0,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.min_expert_capacity = min_expert_capacity
        self.max_group_size = max_group_size
        self.router_z_loss_coef = router_z_loss_coef
        self.router_balance_loss_coef = router_balance_loss_coef
        self.dtype = dtype
        self.training = training
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layer_idx = layer_idx
        self.need_attention_mask = need_attention_mask
        self.is_causal = is_causal
        if init_fn is None:
            init_fn = nnx.initializers.normal(stddev=0.02, dtype=self.dtype)
        
        # Pre-normalization layers (norm before attention and MoE)
        self.attn_norm = DyT(self.d_model, init_alpha=1.0, dtype=self.dtype, rngs=rngs)
        self.moe_norm = DyT(self.d_model, init_alpha=1.0, dtype=self.dtype, rngs=rngs)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            head_dim=self.head_dim,
            dtype=self.dtype,
            training=self.training,
            need_attention_mask=self.need_attention_mask,
            is_causal=self.is_causal,
            init_fn=init_fn,
            rngs=rngs
        )
        
        # MoE feedforward network
        self.feedforward = MixtureLayer(
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
            num_total_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            top_k=self.top_k,
            capacity_factor=self.capacity_factor,
            min_expert_capacity=self.min_expert_capacity,
            max_group_size=self.max_group_size,
            router_z_loss_coef=self.router_z_loss_coef,
            router_balance_loss_coef=self.router_balance_loss_coef,
            dtype=self.dtype,
            training=self.training,
            init_fn=init_fn,
            rngs=rngs
        )
        
    def forward(self, x: jnp.ndarray, attn_mask: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        # Pre-normalization for attention
        attn_norm_x = self.attn_norm(x)
        
        # Attention with residual connection
        if self.use_gradient_checkpointing:
            attn_output = nnx.remat(self.attention)(attn_norm_x, attn_mask)
        else:
            attn_output = self.attention(attn_norm_x, attn_mask)
        x = x + attn_output
        
        # Pre-normalization for MoE
        ff_norm_x = self.moe_norm(x)
        
        # MoE with residual connection
        ff_output, router_loss = self.feedforward(ff_norm_x)
        x = x + ff_output
        
        return x, router_loss
    
    def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        return self.forward(x, attn_mask)

class LLM(nnx.Module):
    def __init__(self,
        d_model: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        vocab_size: int,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2,
        capacity_factor: float = 2.0,
        min_expert_capacity: int = 8,
        max_group_size: int = 4096,
        router_z_loss_coef: float = 1e-3,
        router_balance_loss_coef: float = 1e-4,
        dtype: jnp.dtype = jnp.bfloat16,
        training: bool = False,
        use_gradient_checkpointing: bool = False,
        need_attention_mask: bool = True,
        is_causal: bool = True,
        init_fn = None,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.min_expert_capacity = min_expert_capacity
        self.max_group_size = max_group_size
        self.router_z_loss_coef = router_z_loss_coef
        self.router_balance_loss_coef = router_balance_loss_coef
        self.dtype = dtype
        self.training = training
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.need_attention_mask = need_attention_mask
        self.is_causal = is_causal

        if init_fn is None:
            init_fn = nnx.initializers.normal(stddev=0.02, dtype=self.dtype)
        
        key = rngs.params()
        embedding_key, output_key = jax.random.split(key, 2)
        
        # Token embeddings
        self.token_embedding = nnx.Param(
            init_fn(embedding_key, (self.vocab_size, self.d_model)),
            sharding=(None, None),
            dtype=self.dtype,
        )
        
        # Create transformer blocks
        self.blocks = [
            Block(
                d_model=self.d_model,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_experts=self.num_experts,
                num_shared_experts=self.num_shared_experts,
                top_k=self.top_k,
                capacity_factor=self.capacity_factor,
                min_expert_capacity=self.min_expert_capacity,
                max_group_size=self.max_group_size,
                router_z_loss_coef=self.router_z_loss_coef,
                router_balance_loss_coef=self.router_balance_loss_coef,
                dtype=self.dtype,
                training=self.training,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                need_attention_mask=self.need_attention_mask,
                is_causal=self.is_causal,
                init_fn=init_fn,
                layer_idx=layer_idx,
                rngs=rngs
            ) for layer_idx in range(self.num_layers)
        ]
        
        # Final layer normalization
        self.final_norm = DyT(self.d_model, init_alpha=1.0, dtype=self.dtype, rngs=rngs)
        
        # Output projection
        self.lm_head = nnx.Param(
            init_fn(output_key, (self.d_model, self.vocab_size)),
            sharding=(None, None),
            dtype=self.dtype,
        )
        
    def __call__(self, input_ids: jnp.ndarray, attn_mask: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:

        x = self.token_embedding[input_ids]
        
        # Initialize router loss accumulator only if in training mode
        router_loss = jnp.zeros((), dtype=self.dtype)
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            # During inference, don't track losses at all
            if not self.training:
                x, _ = block(x, attn_mask)
            else:
                x, block_loss = block(x, attn_mask)
                # In training mode, always accumulate losses
                router_loss += block_loss
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = x @ self.lm_head
        
        router_loss /= self.num_layers
        
        return logits, router_loss

class DiffusionLLM(nnx.Module):
    def __init__(self,
        d_model: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        vocab_size: int,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2,
        capacity_factor: float = 2.0,
        min_expert_capacity: int = 8,
        max_group_size: int = 4096,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        router_z_loss_coef: float = 1e-3,
        router_balance_loss_coef: float = 1e-4,
        dtype: jnp.dtype = jnp.bfloat16,
        training: bool = False,
        use_gradient_checkpointing: bool = False,
        need_attention_mask: bool = False,
        init_fn = None,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.min_expert_capacity = min_expert_capacity
        self.max_group_size = max_group_size
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.router_z_loss_coef = router_z_loss_coef
        self.router_balance_loss_coef = router_balance_loss_coef
        self.dtype = dtype
        self.training = training
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.need_attention_mask = need_attention_mask
        

        if init_fn is None:
            init_fn = nnx.initializers.normal(stddev=0.02, dtype=self.dtype)
        
        key = rngs.params()
        
        self.text_head = nnx.Param(
            init_fn(key, (self.vocab_size, self.d_model)),
            sharding=(None, None),
            dtype=self.dtype,
        )
        
        self.blocks = [
            Block(
                d_model=self.d_model,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_experts=self.num_experts,
                num_shared_experts=self.num_shared_experts,
                top_k=self.top_k,
                capacity_factor=self.capacity_factor,
                min_expert_capacity=self.min_expert_capacity,
                max_group_size=self.max_group_size,
                router_z_loss_coef=self.router_z_loss_coef,
                router_balance_loss_coef=self.router_balance_loss_coef,
                dtype=self.dtype,
                training=self.training,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                need_attention_mask=self.need_attention_mask,
                is_causal=False,
                init_fn=init_fn,
                layer_idx=layer_idx,
                rngs=rngs
            ) for layer_idx in range(self.num_layers)
        ]
        
        self.final_norm = DyT(self.d_model, init_alpha=1.0, dtype=self.dtype, rngs=rngs)

        # Build noise schedule
        self.betas = nnx.Variable(jnp.linspace(beta_start, beta_end, timesteps))
        self.alphas = nnx.Variable(1.0 - self.betas)
        self.alphas_cumprod = nnx.Variable(jnp.cumprod(self.alphas.value))
        self.sqrt_alphas_cumprod = nnx.Variable(jnp.sqrt(self.alphas_cumprod.value))
        self.sqrt_one_minus_alphas_cumprod = nnx.Variable(jnp.sqrt(1.0 - self.alphas_cumprod.value))
    
    def encode(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        return self.text_head.value[input_ids]

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum('btd,vd->btv', x, self.text_head.value)

    def noise(self, x_0: jnp.ndarray, t: jnp.ndarray, rngs: nnx.Rngs) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x_0 shape: (batch_size, seq_len, d_model)
        # t shape: (batch_size, seq_len)
        
        noise = jax.random.normal(rngs.params(), shape=x_0.shape, dtype=self.dtype)
        
        # Reshape t to match the dimensions we're working with
        # Add a dimension for d_model
        t = t[..., None]  # shape: (batch_size, seq_len, 1)
        
        sqrt_alphas_cumprod_t = jnp.take(self.sqrt_alphas_cumprod.value, t)
        sqrt_one_minus_alphas_cumprod_t = jnp.take(self.sqrt_one_minus_alphas_cumprod.value, t)
        
        # The broadcasting will now work correctly for token-wise timesteps
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise

    def clamp_embed(self, predicted_x_0: jnp.ndarray) -> jnp.ndarray:
        """Quantize predictions to the nearest token embedding.
        
        Args:
            predicted_x_0: Tensor of shape [batch, seq_len, d_model]
            temperature: Temperature parameter for sampling (higher = more diversity)
            deterministic: If True, use argmax instead of sampling
            rngs: Random number generator state
            
        Returns:
            Quantized embeddings aligned with the token embedding space
        """
        # Find the closest token embedding for each position in predicted_x_0
        # Calculate cosine similarity between predicted_x_0 and all token embeddings
        embeddings = self.text_head.value  # shape: (vocab_size, d_model)
        
        # Normalize embeddings and predicted_x_0 for cosine similarity
        embeddings_norm = embeddings / jnp.linalg.norm(embeddings, axis=1, keepdims=True)
        predicted_x_0_norm = predicted_x_0 / jnp.linalg.norm(predicted_x_0, axis=-1, keepdims=True)
        
        # Calculate similarity scores
        similarity = jnp.einsum('btd,vd->btv', predicted_x_0_norm, embeddings_norm)
        
        top_indices = jnp.argmax(similarity, axis=-1)
        
        # Get the corresponding token embeddings
        quantized_x_0 = self.text_head[top_indices]
        
        return quantized_x_0

    def denoise(self, x_t: jnp.ndarray, t: jnp.ndarray, rngs: nnx.Rngs, attn_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        # x_t shape: (batch_size, seq_len, d_model)
        # t shape: (batch_size, seq_len)
        
        # Reshape t for broadcasting
        t = t[..., None]  # shape: (batch_size, seq_len, 1)
        
        # Get parameters for each token's timestep
        alpha_t = jnp.take(self.alphas.value, t)
        alpha_cumprod_t = jnp.take(self.alphas_cumprod.value, t)
        beta_t = jnp.take(self.betas.value, t)
        
        # Now __call__ predicts x_0 directly instead of noise
        predicted_x_0, _ = self.__call__(x_t, t, attn_mask)

        # Quantize predicted_x_0 to the nearest token embedding
        predicted_x_0 = self.clamp_embed(predicted_x_0)
        
        rng_key = rngs.params()
        
        # Create noise for each token position
        noise = jax.random.normal(rng_key, shape=x_t.shape, dtype=self.dtype)
        
        # Create a mask for t > 0 with proper broadcasting
        t_mask = (t > 0).astype(self.dtype)
        noise = noise * t_mask
        
        # Compute posterior mean using predicted x_0
        posterior_mean = (
            jnp.sqrt(alpha_cumprod_t) * beta_t / (1.0 - alpha_cumprod_t) * predicted_x_0 +
            jnp.sqrt(alpha_t) * (1.0 - alpha_cumprod_t / alpha_t) / (1.0 - alpha_cumprod_t) * x_t
        )
        
        # Compute posterior variance
        posterior_variance = beta_t * (1.0 - alpha_cumprod_t / alpha_t) / (1.0 - alpha_cumprod_t)
        
        # Add a small amount of noise even at timestep 0 to increase diversity
        min_noise_level = 0.001
        noise_scale = jnp.maximum(min_noise_level, jnp.sqrt(posterior_variance))
        
        # Sample from posterior
        x_t_minus_1 = posterior_mean + noise_scale * noise
        
        return x_t_minus_1

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, attn_mask: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Calculate dynamic attention mask based on timesteps
        # t shape: (batch, seq_len)
        # attn_mask shape: (batch, seq_len) - 1 for real tokens, 0 for padding
        
        # Linear interpolation from 1 to 0 based on timestep
        t_ratio = t.astype(self.dtype) / self.timesteps
        dynamic_mask = (1.0 - t_ratio) * (attn_mask if attn_mask is not None else 1.0)
        
        router_loss = jnp.zeros((), dtype=self.dtype)
        for i, block in enumerate(self.blocks):
            if not self.training:
                x, _ = block(x, dynamic_mask)
            else:
                x, block_loss = block(x, dynamic_mask)
                router_loss += block_loss
        
        x = self.final_norm(x)
        router_loss /= self.num_layers
        return x, router_loss

class Test():
    def __init__(self):
        self.batch_size = 1
        self.seq_len = 16
        self.d_model = 512
        self.num_heads = 6
        self.head_dim = 64
        self.hidden_dim = 2048
        self.num_experts = 4
        self.num_shared_experts = 2
        self.num_layers = 2
        self.vocab_size = 32000
        

        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        self.mha = MultiHeadAttention(self.num_heads, self.d_model, self.head_dim, training=True, rngs=rngs)
        self.ff = FeedForward(self.d_model, self.hidden_dim, self.num_experts, training=True, rngs=rngs)
        self.router = Router(self.d_model, self.num_experts, training=True, rngs=rngs)
        self.experts = MixtureLayer(self.d_model, self.hidden_dim, self.num_experts, self.num_shared_experts, training=True, rngs=rngs)
        self.mesh = jax.make_mesh((1, 1), ('expert', 'data'))
        # Add full transformer model
        self.llm = LLM(
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            training=True,
            rngs=rngs
        )

        self.diffusion_llm = DiffusionLLM(
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            training=True,
            rngs=rngs
        )

    @staticmethod
    @nnx.jit
    def test_mha(module, x, attn_mask):
        return module(x, attn_mask)

    @staticmethod
    @nnx.jit
    def test_ff(module, x):
        return module(x)

    @staticmethod
    @nnx.jit
    def test_router(module, x):
        # Use a static expert capacity for jit compilation
        expert_capacity = 16  # Fixed value instead of passing it dynamically
        return module(x, expert_capacity)

    @staticmethod
    @nnx.jit
    def test_experts(module, x):
        return module(x)

    @staticmethod
    @nnx.jit
    def test_transformer(module, input_ids):
        return module(input_ids)
    
    @staticmethod
    @nnx.jit
    def test_diffusion_llm(module: DiffusionLLM, input_ids, rngs: nnx.Rngs, attention_mask: jnp.ndarray):
        x = module.encode(input_ids)
        x_t, noise = module.noise(x, jnp.array([100]), rngs)
        x_t_prev = module.denoise(x_t, jnp.array([99]), rngs, attn_mask=attention_mask)
        logits = module.decode(x_t_prev)
        return logits

    def __call__(self):
        with self.mesh:
            import traceback
            results = {}
            
            # Test MHA
            x_mha = jnp.ones((self.batch_size, self.seq_len, self.d_model))
            attn_mask = jnp.ones((self.batch_size, self.seq_len))
            attn_mask = attn_mask.at[:, self.seq_len - 4:].set(0)
            try:
                y = self.test_mha(self.mha, x_mha, attn_mask)
                print("MHA output shape:", y.shape)
                results['MHA'] = 'SUCCESS'
            except Exception as e:
                print("MHA error:", e)
                print("Stack trace:")
                traceback.print_exc()
                results['MHA'] = 'FAILED'
            
            # Test FF
            x_ff = jnp.ones((self.batch_size, self.num_experts, self.seq_len, self.d_model))
            try:
                y = self.test_ff(self.ff, x_ff)
                print("FF output shape:", y.shape)
                results['FF'] = 'SUCCESS'
            except Exception as e:
                print("FF error:", e)
                print("Stack trace:")
                traceback.print_exc()
                results['FF'] = 'FAILED'
            
            # Test Router
            x_router = jnp.ones((self.batch_size, self.seq_len, self.d_model))
            try:
                a, b, c = self.test_router(self.router, x_router)
                print("Router outputs:", a.shape, b.shape, c)
                results['Router'] = 'SUCCESS'
            except Exception as e:
                print("Router error:", e)
                print("Stack trace:")
                traceback.print_exc()
                results['Router'] = 'FAILED'
            
            # Test Experts
            x_experts = jnp.ones((self.batch_size, self.seq_len, self.d_model))
            try:
                y, z = self.test_experts(self.experts, x_experts)
                print("Experts output shape and loss:", y.shape, z)
                results['Experts'] = 'SUCCESS'
            except Exception as e:
                print("Experts error:", e)
                print("Stack trace:")
                traceback.print_exc()
                results['Experts'] = 'FAILED'
            
            # Test the full transformer
            input_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
            try:
                logits, avg_loss = self.test_transformer(self.llm, input_ids)
                print("Transformer output shape and avg loss:", logits.shape, avg_loss)
                results['Transformer'] = 'SUCCESS'
            except Exception as e:
                print("Transformer error:", e)
                print("Stack trace:")
                traceback.print_exc()
                results['Transformer'] = 'FAILED'
            
            # Test Diffusion LLM
            input_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
            attention_mask = jax.random.uniform(
                jax.random.key(0), 
                shape=(self.batch_size, self.seq_len),
                minval=0.0,
                maxval=1.0
            )
            try:
                rngs = nnx.Rngs(params=jax.random.key(0))
                logits = self.test_diffusion_llm(self.diffusion_llm, input_ids, rngs, attention_mask=attention_mask)
                print("Diffusion LLM output shape:", logits.shape)
                print(logits)
                results['Diffusion LLM'] = 'SUCCESS'
            except Exception as e:
                print("Diffusion LLM error:", e)
                print("Stack trace:")
                traceback.print_exc()
                results['Diffusion LLM'] = 'FAILED'
            
            # Print test summary
            print("\nTest Summary:")
            print("-" * 30)
            for test_name, result in results.items():
                print(f"{test_name:15} | {result}")
            print("-" * 30)
            successes = sum(1 for result in results.values() if result == 'SUCCESS')
            failures = sum(1 for result in results.values() if result == 'FAILED')
            print(f"Total: {len(results)} tests ({successes} passed, {failures} failed)")

if __name__ == "__main__":
    test = Test()
    test()