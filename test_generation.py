import os
import jax
import jax.numpy as jnp
import numpy as np
import time
from transformers import AutoTokenizer
from flax import nnx
if os.path.exists('model.py'):
    from model import Transformer
    from train import MODEL_CONFIG, create_sharded_model
    from config import MESH_SHAPE
#jax.config.update("jax_check_tracer_leaks", True)

MODEL_CONFIG['training'] = False
MODEL_CONFIG['use_gradient_checkpointing'] = False

def make_test_mesh():
    """Create a simple (1,1) mesh for local testing."""
    return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('data', 'expert'))

@nnx.jit
def model_step(model: Transformer, input_ids: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
    """Single forward pass of the model with JIT."""
    logits, _ = model(input_ids, attention_mask)
    return logits

def test_autoregressive_generation():
    # Initialize model and tokenizer
    mesh = make_test_mesh()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Create prompt
    prompt = "Can you tell me"
    input_tokens = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(input_tokens["input_ids"])
    prompt_length = input_ids.shape[1]
    
    with mesh:
        # Create and shard model
        model = create_sharded_model()
        print("Model created and sharded")
        
        # Setup generation
        print("\nGenerating from prompt:", prompt)
        start_time = time.time()
        
        max_new_tokens = 20
        total_length = prompt_length + max_new_tokens
        
        # Pre-fill the sequence with padding tokens
        padded_ids = jnp.pad(
            input_ids,
            ((0, 0), (0, max_new_tokens)),
            mode='constant',
            constant_values=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        
        # Initialize attention mask (1 for prompt tokens, 0 for future tokens)
        attention_mask = jnp.zeros((1, total_length))
        attention_mask = attention_mask.at[:, :prompt_length].set(1)
        
        # Generate tokens
        current_length = prompt_length
        for i in range(max_new_tokens):
            # Forward pass with masked sequence
            logits = model_step(model, padded_ids, attention_mask)
            
            # Get next token (only look at the current position)
            next_token = jnp.argmax(logits[:, current_length-1], axis=-1)
            
            # Update sequence and mask
            padded_ids = padded_ids.at[:, current_length].set(next_token)
            attention_mask = attention_mask.at[:, current_length].set(1)
            current_length += 1
            
            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break
        
        generation_time = time.time() - start_time
        
        # Get the actual generated sequence (without padding)
        generated_ids = padded_ids[:, :current_length]
        generated_text = tokenizer.decode(generated_ids[0])
        
        print(f"\nGenerated text:\n{generated_text}")
        print(f"\nGeneration time: {generation_time:.2f} seconds")
        print(f"Tokens per second: {current_length/generation_time:.2f}")

def main():
    # Run generation test
    test_autoregressive_generation()
    print("\nGeneration test completed successfully! ✨")

if __name__ == "__main__":
    main()