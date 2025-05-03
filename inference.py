import os
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding as NS
import numpy as np
import msgpack
from flax import nnx
from transformers import AutoTokenizer
import argparse

from config import *
from model import LLM
from train_llm import make_mesh, load_checkpoint

@nnx.jit
def _model_generate_step(model: LLM, padded_ids: jnp.ndarray, attention_mask: jnp.ndarray):
    """JIT-compiled single token generation step."""
    logits, _ = model(padded_ids, attention_mask)
    return logits

def sample_with_temperature(logits, temperature=0.0, top_k=0):
    """Sample from logits with temperature and optional top-k filtering."""
    if temperature == 0.0:
        # Greedy sampling
        return jnp.argmax(logits, axis=-1)
    
    # Apply temperature
    logits = logits / jnp.maximum(temperature, 1e-10)
    
    # Optional top-k filtering
    if top_k > 0:
        # Get top-k values and their indices
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        
        # Create a mask for non-top-k values
        mask = jnp.zeros_like(logits)
        mask = mask.at[top_k_indices].set(1)
        
        # Set non-top-k values to large negative number (effectively -inf)
        logits = jnp.where(mask > 0, logits, -1e10)
    
    # Convert to probabilities
    probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    
    # Sample from the distribution
    # Create a deterministic but varying key based on the logits
    seed = int(jnp.sum(jnp.asarray(logits)).item()) % 2**32
    key = jax.random.PRNGKey(seed)
    return jax.random.categorical(key, probs)

def generate_text(model: LLM, tokenizer, prompt: str = "Can you tell me", max_new_tokens: int = 50, 
                  temperature: float = 0.0, top_k: int = 0):
    """Generate text using either greedy or temperature-based sampling."""
    # Tokenize prompt
    input_tokens = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(input_tokens["input_ids"])
    prompt_length = input_ids.shape[1]
    
    # Setup generation
    total_length = prompt_length + max_new_tokens
    
    # Pre-fill sequence with padding
    padded_ids = jnp.pad(
        input_ids,
        ((0, 0), (0, max_new_tokens)),
        mode='constant',
        constant_values=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    
    # Initialize attention mask
    attention_mask = jnp.zeros((1, total_length))
    attention_mask = attention_mask.at[:, :prompt_length].set(1)
    
    # Generate tokens
    current_length = prompt_length
    for _ in range(max_new_tokens):
        # Get next token using JIT-compiled step
        logits = _model_generate_step(model, padded_ids, attention_mask)

        # Sample next token with temperature
        next_token = sample_with_temperature(
            logits[:, current_length-1], 
            temperature=temperature, 
            top_k=top_k
        )
        
        # Update sequence and mask
        padded_ids = padded_ids.at[:, current_length].set(next_token)
        attention_mask = attention_mask.at[:, current_length].set(1)
        current_length += 1
        
        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            break
    
    # Get generated sequence
    generated_ids = padded_ids[:, :current_length]
    generated_text = tokenizer.decode(generated_ids[0])
    
    return generated_text

def setup_inference_model(checkpoint_path=None):
    """Set up the model for inference and load a checkpoint if specified."""
    print("TOTAL DEVICES:", jax.device_count())
    mesh = make_mesh()
    
    with mesh:
        # Set model config to inference mode
        inference_config = MODEL_CONFIG.copy()
        inference_config['training'] = False
        inference_config['use_gradient_checkpointing'] = False
        
        # Create model
        model = LLM(**inference_config, rngs=nnx.Rngs(0))
        print(f"\nCreated model for inference")
        
        # If a specific checkpoint is provided, load it
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                # Call the same checkpoint loading function used in training
                global CHECKPOINT_DIR
                original_checkpoint_dir = CHECKPOINT_DIR
                # Temporarily override checkpoint directory to use the specified file's dir
                CHECKPOINT_DIR = os.path.dirname(checkpoint_path)
                step = load_checkpoint(mesh, model)
                # Restore original checkpoint directory
                CHECKPOINT_DIR = original_checkpoint_dir
                print(f"Loaded checkpoint from step {step}")
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found. Using uninitialized model.")
        else:
            # Load the latest checkpoint
            step = load_checkpoint(mesh, model)
            print(f"Loaded checkpoint from step {step}")
            
        return model, mesh

def inference(prompt, max_tokens=100, temperature=0.0, top_k=0, checkpoint_path=None):
    """Run inference with the model."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # Set up model
    model, mesh = setup_inference_model(checkpoint_path)
    
    # Generate text using our sampling function
    with mesh:
        generated_text = generate_text(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Run inference with the trained model')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0.0 = greedy)')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling (0 = no filtering)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific checkpoint file')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Initialize model once for the interactive session
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        model, mesh = setup_inference_model(args.checkpoint)
        
        print("\n===== Interactive Mode =====")
        print("Type your prompts (or 'exit' to quit)")
        
        while True:
            try:
                prompt = input("\nPrompt> ")
                if prompt.lower() in ('exit', 'quit'):
                    break
                    
                temp_input = input("Temperature (default=0.0): ")
                temperature = float(temp_input) if temp_input.strip() else 0.0
                
                top_k_input = input("Top-k (default=0): ")
                top_k = int(top_k_input) if top_k_input.strip() else 0
                
                with mesh:
                    generated = generate_text(
                        model, 
                        tokenizer, 
                        prompt=prompt, 
                        max_new_tokens=args.max_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                
                print("\nGenerated:")
                print(generated)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    else:
        # Run once with the provided prompt
        generated = inference(
            args.prompt, 
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            checkpoint_path=args.checkpoint
        )
        
        print("\nPrompt:", args.prompt)
        print("\nGenerated:")
        print(generated)

if __name__ == "__main__":
    main() 