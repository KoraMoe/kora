import os
import jax
import jax.numpy as jnp
import msgpack
from flax import nnx
from transformers import AutoTokenizer
import argparse
from typing import Optional

from config import *
from model import LLM
from checkpoint_utils import convert_from_msgpack, load_model_state_from_npz

@nnx.jit
def _model_generate_step(model: LLM, padded_ids: jnp.ndarray, attention_mask: jnp.ndarray):
    """JIT-compiled single token generation step."""
    logits, _ = model(padded_ids, attention_mask)
    return logits

def sample_with_temperature(logits, temperature=0.0, top_k=0, top_p: float = 0.0, rng_key=None):
    """Sample from logits with temperature and optional top-k/top-p filtering."""
    if temperature == 0.0:
        # Greedy sampling
        return jnp.argmax(logits, axis=-1)
    
    if rng_key is None:
        raise ValueError("rng_key must be provided when temperature > 0.")

    # Ensure we're working with float32
    logits = jnp.asarray(logits, dtype=jnp.float32)
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-k filtering if specified
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        # FIXED: Correct unpacking order
        top_k_values, top_k_indices = jax.lax.top_k(logits, top_k)
        
        # Create mask for top-k indices
        topk_mask = jnp.zeros(logits.shape, dtype=bool)
        if logits.ndim == 2:
            batch_indices = jnp.arange(logits.shape[0])[:, None]
            topk_mask = topk_mask.at[batch_indices, top_k_indices].set(True)
        else:  # 1D case
            topk_mask = topk_mask.at[top_k_indices].set(True)
        
        # Set non-top-k logits to -inf
        logits = jnp.where(topk_mask, logits, -jnp.inf)
    
    # Apply top-p (nucleus) filtering if specified
    if 0.0 < top_p < 1.0:
        # Sort probabilities in descending order
        probs = jax.nn.softmax(logits, axis=-1)
        sorted_indices = jnp.argsort(probs, axis=-1)[..., ::-1]
        sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        
        # Create mask for nucleus
        sorted_mask = cumulative_probs <= top_p
        # Ensure at least one token is selected
        if logits.ndim == 2:
            sorted_mask = sorted_mask.at[:, 0].set(True)
        else:
            sorted_mask = sorted_mask.at[0].set(True)
        
        # Map back to original indices
        if logits.ndim == 2:
            batch_indices = jnp.arange(probs.shape[0])[:, None]
            nucleus_mask = jnp.zeros_like(probs, dtype=bool)
            nucleus_mask = nucleus_mask.at[batch_indices, sorted_indices].set(sorted_mask)
        else:
            nucleus_mask = jnp.zeros_like(probs, dtype=bool)
            nucleus_mask = nucleus_mask.at[sorted_indices].set(sorted_mask)
        
        # Apply nucleus mask
        logits = jnp.where(nucleus_mask, logits, -jnp.inf)
    
    # Sample from the distribution
    return jax.random.categorical(rng_key, logits)

def apply_repetition_penalty(
    logits: jnp.ndarray,
    generated_tokens: jnp.ndarray,
    penalty: float,
    pad_token_id: Optional[int] = None,
):
    """Penalize logits for tokens that have already appeared."""
    if penalty == 1.0:
        return logits

    penalty = jnp.asarray(penalty, dtype=logits.dtype)
    mask = jnp.zeros_like(logits, dtype=bool)
    batch_indices = jnp.arange(logits.shape[0])[:, None]
    mask = mask.at[batch_indices, generated_tokens].set(True)

    if pad_token_id is not None:
        mask = mask.at[:, pad_token_id].set(False)

    adjusted_logits = jnp.where(
        mask,
        jnp.where(logits < 0, logits * penalty, logits / penalty),
        logits,
    )
    return adjusted_logits

def generate_text(model: LLM, tokenizer, prompt: str = "Can you tell me", max_new_tokens: int = 50, 
                  temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0, repetition_penalty: float = 1.0,
                  seed: Optional[int] = None):
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
    rng = None
    if temperature > 0.0:
        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="little")
        rng = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        # Get next token using JIT-compiled step
        logits = _model_generate_step(model, padded_ids, attention_mask)
        step_logits = logits[:, current_length-1]
        if repetition_penalty != 1.0:
            step_logits = apply_repetition_penalty(
                step_logits,
                padded_ids[:, :current_length],
                repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Sample next token with temperature
        if temperature == 0.0:
            next_token = sample_with_temperature(
                step_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        else:
            rng, subkey = jax.random.split(rng)
            next_token = sample_with_temperature(
                step_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng_key=subkey
            )
        
        # Update sequence and mask
        padded_ids = padded_ids.at[:, current_length].set(next_token)
        attention_mask = attention_mask.at[:, current_length].set(1)
        current_length += 1
        
        # Check for EOS when tokenizer defines it
        if tokenizer.eos_token_id is not None and int(next_token[0]) == tokenizer.eos_token_id:
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
            extension = os.path.splitext(checkpoint_path)[1].lower()
            model_state_template = nnx.state(model)
            named_sharding = nnx.get_named_sharding(model_state_template, mesh)

            if extension == ".npz":
                pure_state, step = load_model_state_from_npz(checkpoint_path)
                nnx.replace_by_pure_dict(model_state_template, pure_state)
                restored_state = model_state_template
            else:
                with open(checkpoint_path, "rb") as f:
                    try:
                        checkpoint_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
                    except (ValueError, msgpack.exceptions.UnpackException) as e:
                        print(f"Failed to unpack checkpoint {checkpoint_path}: {str(e)}")
                        return 0

                try:
                    model_state_dict = convert_from_msgpack(checkpoint_data["model_state"])
                except KeyError:
                    print(f"Checkpoint {checkpoint_path} missing model_state key")
                    return 0

                if not isinstance(model_state_dict, dict):
                    raise TypeError(f"Expected model_state_dict to be a dictionary, got {type(model_state_dict)}")

                nnx.replace_by_pure_dict(model_state_template, model_state_dict)
                restored_state = model_state_template
                step = checkpoint_data.get("step", 0)

            with mesh:
                sharded_state = jax.tree.map(
                    lambda x, s: jax.device_put(x, s) if isinstance(x, jnp.ndarray) else x,
                    restored_state,
                    named_sharding
                )
                nnx.update(model, sharded_state)

            print(f"Successfully loaded checkpoint from {checkpoint_path}, step {step}")
            return step
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_path}: {str(e)}")
            return 0
    
    # If no direct path or it doesn't exist, try to find the latest checkpoint in CHECKPOINT_DIR
    else:
        checkpoint_dir = CHECKPOINT_DIR  # Use the default from config
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_files = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_") and (f.endswith(".msgpack") or f.endswith(".npz"))
        ]
        
        if not checkpoint_files:
            print(f"No checkpoint found in {checkpoint_dir}, starting from scratch")
            return 0
        
        # Extract step numbers from filenames and find the latest
        steps = [int(f.split("_")[1].split(".")[0]) for f in checkpoint_files]
        steps.sort(reverse=True)  # Sort in descending order to try newest first
        
        latest_step = steps[0]
        # Prefer NPZ if available for the latest step
        candidate_npz = os.path.join(checkpoint_dir, f"checkpoint_{latest_step}.npz")
        if os.path.exists(candidate_npz):
            latest_checkpoint_path = candidate_npz
        else:
            latest_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{latest_step}.msgpack")
        print(f"Found latest checkpoint: {latest_checkpoint_path}")
        
        # Call the function recursively with the specific path
        return load_checkpoint_for_inference(mesh, model, latest_checkpoint_path)

def setup_inference_model(checkpoint_path, preferred_backend: Optional[str] = None):
    """Set up the model for inference and load a checkpoint if specified."""
    if preferred_backend:
        os.environ["JAX_PLATFORM_NAME"] = preferred_backend

    selected_backend = jax.default_backend()
    backend_name = selected_backend.lower()
    print("TOTAL DEVICES:", jax.device_count())
    print("Selected backend:", selected_backend)
    mesh = jax.make_mesh([1, 1], ["data", "expert"])
    
    with mesh:
        # Set model config to inference mode
        inference_config = MODEL_CONFIG.copy()
        inference_config['training'] = False
        inference_config['use_gradient_checkpointing'] = False
        if backend_name == "metal":
            inference_config['dtype'] = jnp.float32
            print("Using float32 dtype for Metal backend.")
        
        # Create model
        model = LLM(**inference_config, rngs=nnx.Rngs(0))
        print(f"\nCreated model for inference")
        
        # Load the checkpoint
        step = load_checkpoint_for_inference(mesh, model, checkpoint_path)
        if step > 0:
            print(f"Model loaded from checkpoint at step {step}")
        else:
            print("Warning: No checkpoint loaded. Using uninitialized model.")
            
        return model, mesh, backend_name

def inference(prompt, max_tokens=100, temperature=0.0, top_k=0, top_p: float = 0.0,
              repetition_penalty: float = 1.0, checkpoint_path=None, seed: Optional[int] = None,
              device: Optional[str] = None):
    """Run inference with the model."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # Set up model
    model, mesh, _ = setup_inference_model(checkpoint_path, preferred_backend=device)
    
    # Generate text using our sampling function
    with mesh:
        generated_text = generate_text(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed
        )

    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Run inference with the trained model')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0.0 = greedy)')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling (0 = no filtering)')
    parser.add_argument('--top_p', type=float, default=0.0, help='Top-p (nucleus) sampling cumulative probability (0 = no filtering)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific checkpoint file')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Penalty for repeated tokens (>1.0 discourages repetition)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for stochastic sampling')
    parser.add_argument('--device', type=str, default=None, help='Preferred JAX backend (e.g., metal, cpu, gpu)')
    
    args = parser.parse_args()

    if args.device:
        os.environ["JAX_PLATFORM_NAME"] = args.device
    
    if args.interactive:
        # Initialize model once for the interactive session
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        model, mesh, backend = setup_inference_model(args.checkpoint, preferred_backend=args.device)
        print(f"Inference using backend: {backend}")
        
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

                top_p_input = input("Top-p (default=0.0): ")
                top_p = float(top_p_input) if top_p_input.strip() else 0.0

                rep_penalty_input = input("Repetition penalty (default=1.0): ")
                repetition_penalty = float(rep_penalty_input) if rep_penalty_input.strip() else 1.0
                
                with mesh:
                    generated = generate_text(
                        model, 
                        tokenizer, 
                        prompt=prompt, 
                        max_new_tokens=args.max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        seed=args.seed
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
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            device=args.device
        )
        
        print("\nPrompt:", args.prompt)
        print("\nGenerated:")
        print(generated)

if __name__ == "__main__":
    main() 
