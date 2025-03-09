import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

@partial(nnx.vmap, in_axes=(0, None, 0))
@partial(nnx.vmap, in_axes=(0, None, None))
def _apply_causal_mask(score_matrix: jnp.ndarray, seq_len: int, attn_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    row_idx = jnp.arange(seq_len)[None, :]
    col_idx = jnp.arange(seq_len)[:, None]
    causal_mask = row_idx <= col_idx
    
    if attn_mask is not None:
        mask = jnp.logical_and(
            causal_mask,
            jnp.logical_or(attn_mask[:, None] > 0, jnp.eye(seq_len, dtype=bool))
        )
    else:
        mask = causal_mask
    
    return jnp.where(mask, score_matrix, -1e9)

class RMSNorm(nnx.Module):
    def __init__(self, 
        dim: int,
        epsilon: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.dim = dim
        self.epsilon = epsilon
        self.dtype = dtype
        
        key = rngs.params()

        self.scale = nnx.Param(
            jax.random.normal(key, (dim,)),
            sharding=(None,),
            dtype=self.dtype,
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Calculate RMS along last dimension
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(variance + self.epsilon)
        
        # Scale and return
        return self.scale * x_norm

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
        init_fn = None,
        rngs: nnx.Rngs = nnx.Rngs()
    ):
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = head_dim
        self.dtype = dtype
        self.training = training
        
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

    def _compute_qkv(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        qkv = jnp.einsum('bsd,tdhm->tbshm', x, self.in_proj.value) # reduce
        q, k, v = qkv

        q, k = self.rotary.rotate_queries_and_keys(q, k)

        q = jnp.einsum('bshd->bhsd', q)
        k = jnp.einsum('bshd->bhsd', k)
        v = jnp.einsum('bshd->bhsd', v)

        return q, k, v
    
    def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape

        q, k, v = self._compute_qkv(x)

        scores = jnp.einsum('bnqd,bnkd->bnqk', q, k) / jnp.sqrt(self.head_dim)
        
        # Handle both causal and attention masking in one step
        if attn_mask is not None and attn_mask.ndim == 2 and attn_mask.shape[0] == batch_size:
            if attn_mask.shape[1] > seq_len:
                attn_mask = attn_mask[:, :seq_len]
        scores = _apply_causal_mask(scores, seq_len, attn_mask)

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
        
        # Input projection weights and bias - store in transposed format for better compute efficiency
        # Store as (d_model, num_experts, hidden_dim) instead of (num_experts, d_model, hidden_dim)
        # This avoids transpose during computation
        self.keys = nnx.Param(
            init_fn(key, (self.d_model, self.num_experts, self.hidden_dim)),
            sharding=(None, 'expert', None),
            dtype=self.dtype,
        )
        self.key_bias = nnx.Param(
            jnp.zeros((self.num_experts, self.hidden_dim)),
            sharding=('expert', None),
            dtype=self.dtype,
        )

        # Output projection weights and bias - store in transposed format
        # Store as (hidden_dim, num_experts, d_model) instead of (num_experts, hidden_dim, d_model)
        self.values = nnx.Param(
            init_fn(key, (self.hidden_dim, self.num_experts, self.d_model)),
            sharding=(None, 'expert', None),
            dtype=self.dtype,
        )
        self.value_bias = nnx.Param(
            jnp.zeros((self.num_experts, self.d_model)),
            sharding=('expert', None),
            dtype=self.dtype,
        )

        self.act = nnx.gelu
    
    def process_by_indices(self, x: jnp.ndarray, indices: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        # indices shape: (batch, seq, top_k)
        # Since weights are now in transposed format, we need to select differently
        # Original shape: keys[e,d,h], values[e,h,d]
        # New shape: keys[d,e,h], values[h,e,d]
        
        # Create index arrays for gathering from transposed weights
        batch_size, seq_len, top_k = indices.shape

        # Prepare indices for gathering
        d_model = x.shape[-1]
        hidden_dim = self.hidden_dim
        
        # Reshape inputs for batched processing
        x_reshaped = x.reshape(-1, d_model)                  # [batch*seq, d_model]
        indices_reshaped = indices.reshape(-1, top_k)        # [batch*seq, top_k]
        weights_reshaped = weights.reshape(-1, top_k)        # [batch*seq, top_k]
        
        # Define a function to process one token with all its top_k experts at once
        def process_token_all_experts(token_vec, expert_indices, token_weights):
            # Function to process a single token-expert pair
            def process_single_expert(expert_idx):
                # Get the expert weights using gather operations instead of dynamic_slice
                # This is more efficient when compiled with XLA
                
                # For forward projection (keys)
                # Extract the expert's weights from the transposed format
                # Using advanced indexing on the expert dimension
                key_weights = self.keys.value[:, expert_idx, :]  # [d_model, hidden_dim]
                
                # Compute hidden representation
                hidden = jnp.dot(token_vec, key_weights)
                hidden = hidden + self.key_bias.value[expert_idx]
                hidden = self.act(hidden)
                
                # For backward projection (values)
                value_weights = self.values.value[:, expert_idx, :]  # [hidden_dim, d_model]
                
                # Compute output
                result = jnp.dot(hidden, value_weights)
                result = result + self.value_bias.value[expert_idx]
                
                return result
            
            # Vectorize across all top_k experts for this token
            # This replaces the outer loop with a single vectorized operation
            results = jax.vmap(process_single_expert)(expert_indices)  # [top_k, d_model]
            
            # Apply weights and sum
            weighted_results = results * token_weights[:, None]  # [top_k, d_model]
            return jnp.sum(weighted_results, axis=0)  # [d_model]
        
        # Vectorize across all batch*seq tokens
        output_flat = jax.vmap(process_token_all_experts)(
            x_reshaped, 
            indices_reshaped, 
            weights_reshaped
        )  # [batch*seq, d_model]
        
        # Reshape back to original format
        output = output_flat.reshape(batch_size, seq_len, d_model)
        
        return output

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden = jax.lax.dot_general(
            x,                    # bsed
            self.keys.value,      # deh (already in optimal format)
            (((3,), (0,)),        # Contract on d dimension 
             ((2,), (1,)))        # Batch dimensions (e batches)
        )
        
        # Add bias to each expert
        hidden = hidden + self.key_bias[None, None, :, :]
        hidden = self.act(hidden)
        
        output = jax.lax.dot_general(
            hidden,               # bseh
            self.values.value,    # hed (already in optimal format)
            (((3,), (0,)),        # Contract on h dimension
             ((2,), (1,)))        # Batch dimensions (e batches)
        )
        
        # Add bias to each expert
        output = output + self.value_bias[None, None, :, :]
        
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
            init_fn = nnx.initializers.normal(stddev=0.02, dtype=self.dtype)

        # Replace direct parameter with linear layer including bias
        self.gate_weight = nnx.Param(
            init_fn(key, (self.d_model, self.num_experts)),
            sharding=(None, None),
            dtype=self.dtype,
        )
        self.gate_bias = nnx.Param(
            jnp.zeros((self.num_experts,)),
            sharding=(None,),
            dtype=self.dtype,
        )

    def __call__(self, x: jnp.ndarray, expert_capacity: int | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        # x shape: (groups, size, d_model)
        # Original: gating_logits = jnp.einsum('gsd,de->gse', x, self.gate_weight.value) + self.gate_bias[None, None, :]

        gating_logits = jax.lax.dot_general(
            x,                        # (groups, size, d_model)
            self.gate_weight.value,   # (d_model, num_experts)
            (((2,), (0,)),            # Contract along d_model dimension
             ((), ()))                # No batched dimensions
        ) + self.gate_bias[None, None, :]
        
        # Compute softmax and top-k in one contiguous block to leverage fusion
        gating_probs = nnx.softmax(gating_logits)
        expert_gate, expert_index = jax.lax.top_k(gating_probs, self.top_k)

        if expert_capacity is None:
            return expert_gate, expert_index, None

        expert_mask = nnx.one_hot(expert_index, num_classes=self.num_experts)

        combined_expert_mask = jnp.sum(expert_mask, axis=2)

        loss = None
        if self.training:
            # Compute both losses together to avoid repeated access to gating data
            router_z_loss = jnp.mean(nnx.logsumexp(gating_logits, axis=-1) ** 2)
            router_balance_loss = jnp.mean(
                jnp.mean(gating_probs, axis=1) * jnp.mean(combined_expert_mask, axis=1)
            ) * (jnp.mean(combined_expert_mask, axis=1).shape[-1] ** 2)
            
            loss = router_balance_loss * self.balance_loss_coef + router_z_loss * self.z_loss_coef

        # Fuse the position calculation with cumsum to enable JAX optimization
        # Use direct implementation of cumulative sum with multiplication
        # which is equivalent to: position_in_expert = jnp.cumsum(expert_mask, axis=1) * expert_mask
        position_in_expert = jnp.cumsum(expert_mask, axis=1) * expert_mask

        valid_assignment = jnp.less(position_in_expert, expert_capacity)
        expert_gate_expanded = expert_gate[..., None]
        
        combine_tensor = jnp.sum(
            expert_gate_expanded[..., None] * 
            valid_assignment.astype(expert_gate.dtype)[..., None] *
            nnx.one_hot(position_in_expert, num_classes=expert_capacity),
            axis=2
        )[..., 1:]
        
        dispatch_mask = combine_tensor.astype(bool)

        return combine_tensor, dispatch_mask, loss

class MixtureLayer(nnx.Module):
    def __init__(self, 
        d_model: int,
        hidden_dim: int,
        num_total_experts: int,
        num_shared_experts: int,
        top_k: int = 2, # theoretically top_k = capacity_factor
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
            init_fn=init_fn,
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
        """
        Compute the optimal group size and expert capacity for routing.
        
        Each token selects top_k experts, so the total expert assignments per group is 
        (group_size * top_k). These assignments need to be distributed across num_experts,
        so the average assignments per expert is (group_size * top_k) / num_experts.
        We multiply by capacity_factor to overprovision and avoid dropped tokens.
        """
        # Convert batch_size and seq_len to integers for static computation
        batch_size_int = int(batch_size)
        seq_len_int = int(seq_len)
        num_tokens = batch_size_int * seq_len_int
        
        # Calculate target group size more efficiently
        # We aim for sqrt(tokens) as a heuristic balancing parallelism and communication
        sqrt_tokens = int(num_tokens ** 0.5)
        target_group_size = min(self.max_group_size, max(32, sqrt_tokens))
        
        # For training, adjust group_size to ensure no padding is needed
        # Find the largest divisor of num_tokens that's <= target_group_size
        # This ensures that num_tokens % group_size == 0
        if self.training:
            # Start with target group size and work downward
            # to find the largest divisor of num_tokens
            group_size = target_group_size
            while group_size > 0:
                if num_tokens % group_size == 0:
                    break
                group_size -= 1
                
            # If we couldn't find a good divisor, use a different approach
            # Find a group_size that's close to target but ensures no padding
            if group_size < 32:  # If we went too small
                # Try finding factors of num_tokens
                factors = []
                for i in range(32, target_group_size + 1):
                    if num_tokens % i == 0:
                        factors.append(i)
                
                if factors:
                    # Choose the largest factor that's closest to our target
                    group_size = max(factors)
                else:
                    # If no good factors, adjust batch or sequence length
                    # Find the closest multiple of target_group_size to num_tokens
                    group_size = target_group_size
                    # Just use original approach - padding is unavoidable
        else:
            group_size = target_group_size
        
        # Calculate number of groups needed (ceil division)
        num_groups = (num_tokens + group_size - 1) // group_size
        
        # When training, we should have exact division (no remainder)
        if self.training:
            num_groups = num_tokens // group_size
            
        # Expert capacity calculation directly tied to top_k
        # Each token chooses top_k experts, so we have (group_size * top_k) total assignments
        # Divided by num_experts gives average assignments per expert
        # Multiplied by capacity_factor for overprovisioning
        expert_capacity = int((group_size * self.top_k * self.capacity_factor) / self.num_total_experts)
        
        # Apply bounds to expert capacity
        min_capacity = max(
            self.min_expert_capacity,  # Absolute minimum
            self.top_k,                # At least top_k to handle worst case of all tokens selecting same expert
            int(group_size * 0.01)     # At least 1% of group size
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
        
        x_expanded = x[:, :, None, :]
        shared_outputs = self.shared_experts(x_expanded)  # [batch, seq, num_shared_experts, d_model]
        output = jnp.mean(shared_outputs, axis=2)  # [batch, seq, d_model]

        return output
    
    def _group_inputs(self, x: jnp.ndarray) -> tuple[jnp.ndarray, int, tuple[int, int], bool]:
        # Extract static dimensions where possible to help JAX optimization
        original_batch_size, original_seq_len, d_model = x.shape
        
        # Calculate group parameters - keep these calculations static where possible
        # to enable better XLA fusion
        group_size, num_groups, expert_capacity = self._compute_group_size(original_batch_size, original_seq_len)
        
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
        # Step 1: Group inputs once and avoid repeated operations
        x_grouped, expert_capacity, original_shape, was_padded = self._group_inputs(x)
        
        # Step 2: Cache shared expert output computation while router is running
        # This enables parallel computation of shared experts and router logic
        shared_output = self._process_shared_experts(x_grouped)
        
        # Step 3: Get routing information in single router call to minimize communication
        combine_tensor, dispatch_mask, router_loss = self.router(x_grouped, expert_capacity)
        
        # initialize output with shared experts result
        output = shared_output

        if self.num_ff_experts > 0:
            ff_dispatch = dispatch_mask[:, :, :self.num_ff_experts, :]
            ff_combine = combine_tensor[:, :, :self.num_ff_experts, :]

            expert_inputs = jax.lax.dot_general(
                ff_dispatch,            # [G, S, E, C]
                x_grouped[:, :, None, :],  # [G, S, 1, d]
                (((3,), (0,)),          # Contract on token capacity dimension (C) with a dummy dim
                 ((0, 1), (0, 1)))      # Batch dimensions (G, S)
            )
            # Result shape: [G, S, E, d]
            
            # Process all feedforward experts in parallel
            expert_outputs = self.feedforward_experts(expert_inputs)
            # Original: ff_output = jnp.einsum('GSEd,GSEC->GSd', expert_outputs, ff_combine)
            ff_output = jax.lax.dot_general(
                expert_outputs,         # [G, S, E, d]
                ff_combine[:, :, :, :, None],  # [G, S, E, C, 1]
                (((2, 3), (2, 4)),      # Contract on E and d/dummy dimensions
                 ((0, 1), (0, 1)))      # Batch dimensions (G, S)
            )
            # Result shape: [G, S, C] - need to sum over C
            ff_output = jnp.sum(ff_output, axis=2)
            
            output = output + ff_output
        
        # Final reshape to original dimensions
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
        
        if init_fn is None:
            init_fn = nnx.initializers.normal(stddev=0.02, dtype=self.dtype)
        
        # Pre-normalization layers (norm before attention and MoE)
        self.attn_norm = RMSNorm(self.d_model, epsilon=1e-6, dtype=self.dtype, rngs=rngs)
        self.moe_norm = RMSNorm(self.d_model, epsilon=1e-6, dtype=self.dtype, rngs=rngs)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            head_dim=self.head_dim,
            dtype=self.dtype,
            training=self.training,
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
        attn_output = self.attention(attn_norm_x, attn_mask)
        x = x + attn_output
        
        # Pre-normalization for MoE
        ff_norm_x = self.moe_norm(x)
        
        # MoE with residual connection
        ff_output, router_loss = self.feedforward(ff_norm_x)
        x = x + ff_output
        
        return x, router_loss
    
    def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        if self.use_gradient_checkpointing:
            return nnx.remat(self.forward)(x, attn_mask)
        return self.forward(x, attn_mask)

class Transformer(nnx.Module):
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
                init_fn=init_fn,
                layer_idx=layer_idx,
                rngs=rngs
            ) for layer_idx in range(self.num_layers)
        ]
        
        # Final layer normalization
        self.final_norm = RMSNorm(self.d_model, epsilon=1e-6, dtype=self.dtype, rngs=rngs)
        
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
        
        # Add full transformer model
        self.transformer = Transformer(
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
        expert_capacity = 4  # Fixed value instead of passing it dynamically
        return module(x, expert_capacity)

    @staticmethod
    @nnx.jit
    def test_experts(module, x):
        return module(x)

    @staticmethod
    @nnx.jit
    def test_transformer(module, input_ids):
        return module(input_ids)

    def __call__(self):
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
            results['MHA'] = 'FAILED'
        
        # Test FF
        x_ff = jnp.ones((self.batch_size, self.seq_len, self.num_experts, self.d_model))
        try:
            y = self.test_ff(self.ff, x_ff)
            print("FF output shape:", y.shape)
            results['FF'] = 'SUCCESS'
        except Exception as e:
            print("FF error:", e)
            results['FF'] = 'FAILED'
        
        # Test Router
        x_router = jnp.ones((self.batch_size, self.seq_len, self.d_model))
        try:
            a, b, c = self.test_router(self.router, x_router)  # Removed expert_capacity argument
            print("Router outputs:", a.shape, b.shape, c)
            results['Router'] = 'SUCCESS'
        except Exception as e:
            print("Router error:", e)
            results['Router'] = 'FAILED'
        
        # Test Experts
        x_experts = jnp.ones((self.batch_size, self.seq_len, self.d_model))
        try:
            y, z = self.test_experts(self.experts, x_experts)
            print("Experts output shape and loss:", y.shape, z)
            results['Experts'] = 'SUCCESS'
        except Exception as e:
            print("Experts error:", e)
            results['Experts'] = 'FAILED'
        
        # Test the full transformer
        input_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
        try:
            logits, avg_loss = self.test_transformer(self.transformer, input_ids)
            print("Transformer output shape and avg loss:", logits.shape, avg_loss)
            results['Transformer'] = 'SUCCESS'
        except Exception as e:
            print("Transformer error:", e)
            results['Transformer'] = 'FAILED'
        
        # Display module info
        print("\nModule details:")
        nnx.display(self.mha)
        nnx.display(self.ff)
        nnx.display(self.router)
        nnx.display(self.experts)
        nnx.display(self.transformer)
        
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