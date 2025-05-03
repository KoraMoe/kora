import os
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding as NS
import numpy as np
import msgpack
from flax import nnx
from transformers import AutoTokenizer
import argparse
from typing import Optional

from config import *
from model import LLM

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

def load_checkpoint_for_inference(
    mesh: jax.sharding.Mesh,
    model: LLM,
    checkpoint_path: Optional[str] = None
) -> int:
    """
    Load a checkpoint directly from a specified path for inference.
    If no path is specified, try to find the latest checkpoint in CHECKPOINT_DIR.
    
    Args:
        mesh: JAX mesh for model distribution
        model: The model to load the checkpoint into
        checkpoint_path: Direct path to the checkpoint file (.msgpack)
        
    Returns:
        The step number of the loaded checkpoint, or 0 if none was loaded
    """
    # If checkpoint_path is provided, use it directly
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            # Load checkpoint data using msgpack
            with open(checkpoint_path, "rb") as f:
                try:
                    checkpoint_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
                except (ValueError, msgpack.exceptions.UnpackException) as e:
                    print(f"Failed to unpack checkpoint {checkpoint_path}: {str(e)}")
                    return 0
            
            # Convert structured arrays back to appropriate types
            def convert_from_msgpack(obj):
                if isinstance(obj, dict):
                    # Check if this is a serialized array
                    if "__jax_array__" in obj:
                        # Handle the case where array data might be None due to serialization issues
                        if obj["data"] is None:
                            print(f"Warning: Found array with missing data. Error: {obj.get('error', 'Unknown')}")
                            # Create an empty array with the right shape and dtype if possible
                            shape = obj.get("shape")
                            dtype_str = obj.get("dtype", "float32")
                            if shape is not None:
                                return jnp.zeros(shape, dtype=dtype_str)
                            else:
                                # If we don't have shape info, return a scalar zero
                                return jnp.array(0, dtype=dtype_str)
                        # Normal case - convert back to JAX array
                        try:
                            array_data = np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
                            return jnp.array(array_data)
                        except Exception as array_error:
                            print(f"Error converting array: {array_error}")
                            # Fallback to zeros with appropriate shape
                            shape = obj.get("shape")
                            dtype_str = obj.get("dtype", "float32")
                            if shape is not None:
                                return jnp.zeros(shape, dtype=dtype_str)
                            else:
                                return jnp.array(0, dtype=dtype_str)
                    elif "__numpy_array__" in obj:
                        # Convert back to NumPy array
                        try:
                            return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
                        except Exception as np_error:
                            print(f"Error converting numpy array: {np_error}")
                            shape = obj.get("shape")
                            dtype_str = obj.get("dtype", "float32")
                            if shape is not None:
                                return np.zeros(shape, dtype=dtype_str)
                            else:
                                return np.array(0, dtype=dtype_str)
                    else:
                        # Regular dictionary
                        return {k: convert_from_msgpack(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_from_msgpack(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_from_msgpack(item) for item in obj)
                else:
                    return obj
            
            try:
                # Create abstract model state
                model_state = nnx.state(model)
                # Get the named sharding for the model based on the mesh
                named_sharding = nnx.get_named_sharding(model_state, mesh)

                # Process model state
                model_state_dict = convert_from_msgpack(checkpoint_data["model_state"])

                # Ensure we have a dictionary type for the state
                if not isinstance(model_state_dict, dict):
                    raise TypeError(f"Expected model_state_dict to be a dictionary, got {type(model_state_dict)}")

                # Replace abstract state with restored state
                nnx.replace_by_pure_dict(model_state, model_state_dict)

                # Apply sharding constraints to the state to ensure it's properly distributed
                with mesh:
                    # Use jax.device_put with tree_map to apply sharding to each array in the state
                    sharded_state = jax.tree.map(
                        lambda x, s: jax.device_put(x, s) if isinstance(x, jnp.ndarray) else x,
                        model_state, named_sharding
                    )
                    
                    # Update the model with the sharded state
                    nnx.update(model, sharded_state)
                
                # Get the step from the checkpoint data
                step = checkpoint_data.get("step", 0)
                print(f"Successfully loaded checkpoint from {checkpoint_path}, step {step}")
                return step
                
            except Exception as e:
                print(f"Error processing checkpoint {checkpoint_path}: {str(e)}")
                return 0
                
        except Exception as e:
            print(f"Error opening checkpoint file {checkpoint_path}: {str(e)}")
            return 0
    
    # If no direct path or it doesn't exist, try to find the latest checkpoint in CHECKPOINT_DIR
    else:
        checkpoint_dir = CHECKPOINT_DIR  # Use the default from config
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".msgpack")]
        
        if not checkpoint_files:
            print(f"No checkpoint found in {checkpoint_dir}, starting from scratch")
            return 0
        
        # Extract step numbers from filenames and find the latest
        steps = [int(f.split("_")[1].split(".")[0]) for f in checkpoint_files]
        steps.sort(reverse=True)  # Sort in descending order to try newest first
        
        latest_step = steps[0]
        latest_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{latest_step}.msgpack")
        print(f"Found latest checkpoint: {latest_checkpoint_path}")
        
        # Call the function recursively with the specific path
        return load_checkpoint_for_inference(mesh, model, latest_checkpoint_path)

def setup_inference_model(checkpoint_path):
    """Set up the model for inference and load a checkpoint if specified."""
    print("TOTAL DEVICES:", jax.device_count())
    mesh = jax.make_mesh([1, 1], ["data", "expert"])
    
    with mesh:
        # Set model config to inference mode
        inference_config = MODEL_CONFIG.copy()
        inference_config['training'] = False
        inference_config['use_gradient_checkpointing'] = False
        
        # Create model
        model = LLM(**inference_config, rngs=nnx.Rngs(0))
        print(f"\nCreated model for inference")
        
        # Load the checkpoint
        step = load_checkpoint_for_inference(mesh, model, checkpoint_path)
        if step > 0:
            print(f"Model loaded from checkpoint at step {step}")
        else:
            print("Warning: No checkpoint loaded. Using uninitialized model.")
            
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