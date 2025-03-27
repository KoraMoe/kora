import os
import wandb
import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as PS, NamedSharding as NS
import queue
import threading

from flax import nnx
import optax

from model import DiffusionLLM
from tqdm import tqdm
from datasets import load_from_disk
import orbax.checkpoint as ocp
from typing import Dict, Any, Optional, Tuple
from jax.experimental.multihost_utils import sync_global_devices
from transformers import AutoTokenizer
from config import *

class BatchLoader:
    def __init__(self, dataset, batch_size: int, data_spec: NS, seed: int = 42, num_prefetch: int = 6):
        self.dataset = dataset
        self.global_batch_size = batch_size
        self.data_spec = data_spec
        self.base_seed = seed
        self.num_prefetch = num_prefetch
        
        # Process-specific information
        self.process_id = jax.process_index()
        self.num_processes = jax.process_count()
        
        # Calculate per-process batch size
        assert self.global_batch_size % self.num_processes == 0, "Batch size must be divisible by number of processes"
        self.local_batch_size = self.global_batch_size // self.num_processes
        
        # Calculate number of samples and steps
        self.num_samples = len(dataset)
        self.steps_per_epoch = self.num_samples // self.global_batch_size
        
        # Initialize state
        self.current_epoch = 0
        self.current_idx = 0
        
        # Initial shuffle with base seed
        self._shuffle_indices()
        
        # Initialize threading components
        self._queue = queue.Queue(maxsize=num_prefetch)
        self._stop_event = threading.Event()
        self._thread_lock = threading.Lock()
        self._start_producer_thread()
    
    def _start_producer_thread(self):
        """Start a new producer thread with proper locking."""
        with self._thread_lock:
            # Stop existing thread if running
            self._stop_existing_thread()
            
            # Clear the queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            
            # Reset stop event
            self._stop_event.clear()
            
            # Start new thread
            self._thread = threading.Thread(target=self._producer_thread, daemon=True)
            self._thread.start()
    
    def _stop_existing_thread(self):
        """Safely stop the existing producer thread."""
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=5.0)  # Wait up to 5 seconds
            if self._thread.is_alive():
                print("Warning: Producer thread did not stop gracefully")
    
    def restart(self):
        """Safely restart the producer thread."""
        with self._thread_lock:
            self._stop_existing_thread()
            self._start_producer_thread()
    
    def _create_batch(self, indices):
        """Create a sharded batch from indices for the current process."""
        examples = {}
        # Ensure indices are valid and within range
        valid_indices = np.clip(indices, 0, len(self.dataset) - 1)
        batch_data = self.dataset[valid_indices]
        
        for k, v in batch_data.items():
            array = np.array(v, dtype=jnp.int32)
            examples[k] = jax.make_array_from_process_local_data(
                self.data_spec,
                array
            )
        
        return examples
    
    def _shuffle_indices(self):
        """Shuffle dataset indices using deterministic seed based on epoch."""
        epoch_seed = self.base_seed + self.current_epoch
        rng = np.random.RandomState(seed=epoch_seed)
        
        # Shuffle all indices first
        all_indices = rng.permutation(self.num_samples)
        
        # Calculate process-specific shard of indices
        shard_size = len(all_indices) // self.num_processes
        start_idx = self.process_id * shard_size
        end_idx = start_idx + shard_size
        
        # Get process-specific indices
        self.indices = all_indices[start_idx:end_idx]
        self.current_idx = 0
        self.current_epoch += 1
    
    def _producer_thread(self):
        """Producer thread that continuously generates batches and puts them in the queue."""
        try:
            while not self._stop_event.is_set():
                if self.current_idx + self.local_batch_size > len(self.indices):
                    self._shuffle_indices()
                
                # Get local batch indices for this process
                batch_indices = self.indices[self.current_idx:self.current_idx + self.local_batch_size]
                self.current_idx += self.local_batch_size
                
                # Create batch and put in queue
                batch = self._create_batch(batch_indices)
                
                # Check stop event before potentially blocking on queue put
                if self._stop_event.is_set():
                    break
                    
                self._queue.put(batch)
        except Exception as e:
            import traceback
            print(f"Error in producer thread: {str(e)}")
            print(traceback.format_exc())
            self._stop_event.set()
    
    def next(self):
        """Get next batch from the queue."""
        if self._stop_event.is_set():
            # Attempt to restart the thread if it died
            print("Producer thread died, attempting restart...")
            self.restart()
            
        try:
            # Use a timeout to prevent indefinite blocking
            batch = self._queue.get(timeout=30.0)
            return batch
        except queue.Empty:
            print("Queue timeout, restarting producer thread...")
            self.restart()
            # Try one more time
            return self._queue.get(timeout=30.0)
    
    def __del__(self):
        """Cleanup when the loader is destroyed."""
        self._stop_existing_thread()

def load_dataset():
    # Load the tokenized dataset
    dataset = load_from_disk(TOKENIZED_DATASET_PATH)

    # Use indices to create test set
    dataset_size = len(dataset)  # type: ignore
    test_size = BATCH_SIZE
    
    all_indices = list(range(dataset_size))
    test_indices = all_indices[-test_size:]
    train_indices = all_indices[:-test_size]
    
    # Split using the indices
    train_dataset = dataset.select(train_indices)  # type: ignore
    test_dataset = dataset.select(test_indices)  # type: ignore
    
    return train_dataset, test_dataset

def make_mesh():
    mesh = jax.make_mesh((4, 1), ("data", "expert"))
    return mesh

@nnx.jit
def create_sharded_model():
    model = DiffusionLLM(**MODEL_CONFIG, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model

def create_learning_rate_schedule(
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.1
) -> optax.Schedule:
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps
    )
    
    decay_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=min_lr_ratio
    )
    
    return optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_steps]
    )


def count_params(model: DiffusionLLM) -> float:
    """Count total number of trainable parameters in billions."""
    total = 0
    state = nnx.state(model)
    for param in jax.tree_util.tree_leaves(state):
        if isinstance(param, jnp.ndarray):
            total += param.size
    return total / 1e9  # Convert to billions

def create_checkpoint_manager(base_dir: str = "checkpoints", max_to_keep: int = 5) -> ocp.CheckpointManager:
    """Create an Orbax checkpoint manager with the new API style."""
    os.makedirs(base_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=True,
        enable_async_checkpointing=True,
    )
    # Use the new API style without additional parameters
    return ocp.CheckpointManager(
        base_dir,
        options=options,
    )

def save_checkpoint(
    ckpt_manager: ocp.CheckpointManager, 
    model: DiffusionLLM, 
    optimizer: nnx.Optimizer, 
    step: int,
    batch_loader: BatchLoader,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Save the model, optimizer state, current step, and batch state to checkpoint using new API."""
    # Get states to save
    model_state = nnx.state(model)
    optimizer_state = nnx.state(optimizer)
    
    # Create batch state dict to save batch loader state
    batch_state = {
        "current_epoch": batch_loader.current_epoch,
        "current_idx": batch_loader.current_idx,
        "indices": np.array(batch_loader.indices),  # Convert to numpy for serialization
    }
    
    ckpt_manager.save(
        step, 
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(model_state),
            optimizer=ocp.args.StandardSave(optimizer_state),
            batch_state=ocp.args.StandardSave(batch_state),
            model_stats=ocp.args.StandardSave(metrics or {})
        )
    )
    ckpt_manager.wait_until_finished()
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(
    ckpt_manager: ocp.CheckpointManager,
    model: DiffusionLLM,
    optimizer: nnx.Optimizer,
    batch_loader: Optional[BatchLoader] = None
) -> Tuple[int, Dict[str, Any]]:
    """Load checkpoint into existing model and optimizer if available using new API."""
    # Check if checkpoint exists
    step = ckpt_manager.latest_step()
    if step is None:
        print("No checkpoint found, starting from scratch")
        return 0, {}
    
    # Create abstract target based on the existing model and optimizer
    abs_model_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if hasattr(x, 'shape') else x,
        nnx.state(model)
    )
    
    abs_optimizer_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if hasattr(x, 'shape') else x,
        nnx.state(optimizer)
    )

    # Restore checkpoint
    print(f"Restoring checkpoint from step {step}")
    restored = ckpt_manager.restore(
        step, 
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(abs_model_state),
            optimizer=ocp.args.StandardRestore(abs_optimizer_state),
            batch_state=ocp.args.StandardRestore(),
            model_stats=ocp.args.StandardRestore()
        )
    )
    
    # Update model and optimizer states
    nnx.state(model).update(restored["model"])
    nnx.state(optimizer).update(restored["optimizer"])
    
    # Restore batch loader state if provided
    if batch_loader is not None:
        try:
            batch_state = restored["batch_state"]
            batch_loader.current_epoch = batch_state["current_epoch"]
            batch_loader.current_idx = batch_state["current_idx"]
            
            # Handle potential issues with indices
            indices = batch_state["indices"]
            if len(indices) > 0:
                # Ensure indices are within the valid range for the dataset
                indices = np.clip(indices, 0, len(batch_loader.dataset) - 1)
                batch_loader.indices = indices
            else:
                batch_loader.indices = np.random.RandomState(seed=batch_loader.base_seed).permutation(batch_loader.num_samples)
            
            # Restart the batch loader thread with new state
            batch_loader.restart()
            
        except Exception as e:
            print(f"Warning: Failed to restore batch loader state: {str(e)}")
            print("Reinitializing batch loader...")
            # Reset the batch loader to initial state
            batch_loader.current_epoch = 0
            batch_loader.current_idx = 0
            batch_loader.indices = np.random.RandomState(seed=batch_loader.base_seed).permutation(batch_loader.num_samples)
            batch_loader.restart()
    
    return step, restored.get("model_stats", {})

@nnx.jit
def train_step(model: DiffusionLLM, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
    key = rngs.params()

    def loss_fn(model: DiffusionLLM):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']  # 1 for real tokens, 0 for padding
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        t = jax.random.randint(
            key, 
            shape=(batch_size, seq_len),
            minval=0, 
            maxval=model.timesteps, 
            dtype=jnp.int32
        )

        x_0 = model.encode(input_ids)
        x_t, _ = model.noise(x_0, t, rngs)

        # Model now predicts x_0 directly
        predicted_x_0, router_loss = model(x_t, t, attn_mask=attention_mask)
        
        # Decode predicted x_0 to get logits over vocabulary
        logits = model.decode(predicted_x_0)  # shape: [batch_size, seq_len, vocab_size]

        # Create loss mask: 1 for t > 0 and valid tokens, 0 for t = 0 or padding
        loss_mask = (t > 0).astype(jnp.float32) * attention_mask
        
        # Compute cross entropy loss
        # Using standard cross entropy with optional label smoothing
        labels = nnx.one_hot(input_ids, num_classes=logits.shape[-1])
        per_token_loss = -jnp.sum(
            labels * nnx.log_softmax(logits),
            axis=-1
        )
        
        # Apply loss mask and compute mean loss
        masked_loss = jnp.sum(
            loss_mask * per_token_loss
        ) / (jnp.sum(loss_mask) + 1e-8)

        total_loss = masked_loss + jnp.mean(router_loss)
        
        return total_loss, (masked_loss, router_loss)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (x0_loss, router_loss)), grads = grad_fn(model)
    
    # Update model parameters
    optimizer.update(grads)
    
    # Update metrics
    metrics.update(
        total_loss=total_loss,
        x0_loss=x0_loss,
        router_loss=router_loss
    )

    return total_loss

@nnx.jit
def eval_step(model: DiffusionLLM, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
    key = rngs.params()

    def loss_fn(model: DiffusionLLM):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        t = jax.random.randint(
            key, 
            shape=(batch_size, seq_len),
            minval=0, 
            maxval=model.timesteps, 
            dtype=jnp.int32
        )

        x_0 = model.encode(input_ids)
        x_t, noise = model.noise(x_0, t, rngs)

        predicted_noise, router_loss = model(x_t, t, attn_mask=attention_mask)

        # Create loss mask: 1 for t > 0 and valid tokens, 0 for t = 0 or padding
        loss_mask = (t > 0).astype(predicted_noise.dtype)[..., None] * attention_mask[..., None]
        
        # Only compute loss for non-zero timesteps and valid tokens
        noise_loss = jnp.sum(
            loss_mask * ((predicted_noise - noise) ** 2)
        ) / (jnp.sum(loss_mask) + 1e-8)  # Add small epsilon to avoid division by zero

        total_loss = noise_loss + jnp.mean(router_loss)
        
        return total_loss, (noise_loss, router_loss)
    
    total_loss, (noise_loss, router_loss) = loss_fn(model)
    
    # Update metrics
    metrics.update(
        total_loss=total_loss,
        noise_loss=noise_loss,
        router_loss=router_loss
    )

def sample_text(model: DiffusionLLM, tokenizer, prompt: str = "Can you tell me", seq_len: int = 32, rngs: nnx.Rngs = nnx.Rngs(0)):
    @nnx.jit
    def encode_input(model, input_ids):
        return model.encode(input_ids)

    @nnx.jit
    def add_noise(model, x, t, rngs):
        return model.noise(x, t, rngs)

    @nnx.jit
    def denoise_step(model, x_t, t, rngs, attention_mask, prompt_mask, x_0):
        # Denoise
        # Add extra dimension to attention_mask to match expected shape
        attention_mask = attention_mask[..., None]  # Shape becomes (batch, seq_len, 1)
        new_x = model.denoise(x_t, t, rngs, attention_mask)
        # Keep prompt tokens unchanged
        return jnp.where(prompt_mask[..., None], x_0, new_x)

    @nnx.jit
    def decode_output(model, x_t):
        logits = model.decode(x_t)
        return jnp.argmax(logits, axis=-1)

    # Tokenize prompt
    input_tokens = tokenizer(prompt, return_tensors="np", padding=False, truncation=True)
    input_ids = jnp.array(input_tokens["input_ids"])
    prompt_len = input_ids.shape[1]
    
    # Pad or truncate to fixed sequence length
    padded_input_ids = jnp.zeros((1, seq_len), dtype=jnp.int32)
    padded_input_ids = padded_input_ids.at[:, :prompt_len].set(input_ids)
    
    # Create prompt mask (1 for prompt tokens, 0 for noise tokens)
    prompt_mask = jnp.zeros((1, seq_len), dtype=jnp.bool_)
    prompt_mask = prompt_mask.at[:, :prompt_len].set(True)
    
    # Pad the batch dimension to match data parallel size (4)
    input_ids = jnp.tile(padded_input_ids, (4, 1))
    prompt_mask = jnp.tile(prompt_mask, (4, 1))
    attention_mask = jnp.ones_like(input_ids)  # All ones for full attention
    
    # Encode prompt
    x_0 = encode_input(model, input_ids)
    
    # Initialize timesteps - 0 for prompt, max timestep for noise tokens
    t = jnp.where(
        prompt_mask,
        jnp.zeros_like(input_ids),
        jnp.full_like(input_ids, model.timesteps - 1)
    )
    
    # Add noise only to non-prompt positions
    x_t, _ = add_noise(model, x_0, t, rngs)
    
    # Only use noise for non-prompt positions
    x_t = jnp.where(prompt_mask[..., None], x_0, x_t)
    
    # Denoise step by step
    for timestep in range(model.timesteps - 1, -1, -1):
        # Set timestep - keep 0 for prompt tokens, current timestep for noise tokens
        t = jnp.where(
            prompt_mask,
            jnp.zeros_like(input_ids),
            jnp.full_like(input_ids, timestep)
        )
        x_t = denoise_step(model, x_t, t, rngs, attention_mask, prompt_mask, x_0)
    
    # Decode to tokens
    generated_ids = decode_output(model, x_t)
    
    # Only take the first sequence since we tiled the input
    generated_text = tokenizer.decode(generated_ids[0])
    return generated_text

def main():
    train_dataset, test_dataset = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    if jax.process_index() == 0:
        wandb.init(
            project="kora-diffusion",
            config={
                "context_length": CONTEXT_LENGTH,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "warmup_steps": WARMUP_STEPS,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "dtype": str(DTYPE),
                "model_config": MODEL_CONFIG,
            }
        )
    
    print(f"Process {jax.process_index()}: Initialized distributed runtime")
    sync_global_devices("distributed_init")

    mesh = make_mesh()
    data_spec = NS(mesh, PS('data', None))

    # Create batch loaders
    train_loader = BatchLoader(dataset=train_dataset, batch_size=BATCH_SIZE, data_spec=data_spec)
    test_loader = BatchLoader(dataset=test_dataset, batch_size=BATCH_SIZE, data_spec=data_spec, seed=43)
    print(f"Process {jax.process_index()}: Created data loaders")
    sync_global_devices("data_loaders_created")

    # Create checkpoint manager
    ckpt_manager = create_checkpoint_manager(CHECKPOINT_DIR)
    
    # Calculate training steps
    num_epochs = NUM_EPOCHS
    steps_per_epoch = train_loader.steps_per_epoch
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = min(2000, total_steps // 10)

    with mesh:
        # Create model
        model = create_sharded_model()
        print(f"Process {jax.process_index()}: Created sharded model")
        sync_global_devices("model_created")
        
        # Print model size
        print(f"\nModel Parameters: {count_params(model):.2f}B")
        print("-" * 50)

        # Create learning rate schedule and optimizer
        lr_schedule = create_learning_rate_schedule(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=LEARNING_RATE
        )

        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(GRADIENT_CLIP_NORM),
                optax.adamw(
                    learning_rate=lr_schedule,
                    b1=0.9,
                    b2=0.95,
                    eps=1e-8,
                    weight_decay=0.1,
                )
            )
        )
        print(f"Process {jax.process_index()}: Created optimizer")
        sync_global_devices("optimizer_created")
        
        # Try to restore from checkpoint
        start_step, metrics = load_checkpoint(ckpt_manager, model, optimizer, train_loader)
        print(f"Process {jax.process_index()}: Restored checkpoint state")
        sync_global_devices("checkpoint_restored")
        
        # Initialize best eval loss from checkpoint metrics or default
        best_eval_loss = metrics.get("best_eval_loss", float('inf'))
        if start_step > 0:
            print(f"Process {jax.process_index()}: Resuming from step {start_step} with best eval loss: {best_eval_loss:.4f}")
        
        # Create metrics tracker
        train_metrics = nnx.MultiMetric(
            total_loss=nnx.metrics.Average('total_loss'),
            x0_loss=nnx.metrics.Average('x0_loss'),
            router_loss=nnx.metrics.Average('router_loss')
        )
        
        eval_metrics = nnx.MultiMetric(
            total_loss=nnx.metrics.Average('total_loss'),
            noise_loss=nnx.metrics.Average('noise_loss'),
            router_loss=nnx.metrics.Average('router_loss')
        )

        # Main training loop
        progress_bar = tqdm(total=total_steps - start_step, desc=f"Training")
        for step in range(start_step, total_steps):
            batch = train_loader.next()
            rngs = nnx.Rngs(params=jax.random.PRNGKey(step))

            train_step(model, optimizer, train_metrics, rngs, batch)
            progress_bar.update(1)
            
            if step % SAMPLE_STEPS == 0:
                print(f"\nSampling text at step {step + 1}:")
                generated_text = sample_text(model, tokenizer, rngs=rngs)
                print(f"Generated: {generated_text}\n")
                if jax.process_index() == 0:
                    wandb.log({
                        "generated_text": generated_text,
                        "step": step
                    })
            
            if (step + 1) % LOG_STEPS == 0 or step == total_steps - 1:
                metrics_values = train_metrics.compute()
                train_metrics.reset()
                
                progress_bar.set_postfix({
                    'loss': f"{metrics_values['total_loss']:.4f}",
                    'x0_loss': f"{metrics_values['x0_loss']:.4f}",
                    'lr': f"{lr_schedule(step):.6f}"
                })

                if jax.process_index() == 0:
                    wandb.log({
                        f"train/{k}": v for k, v in metrics_values.items()
                    } | {
                        'train/step': step,
                        'train/epoch': step / steps_per_epoch
                    })
            
            # Evaluate and save checkpoint periodically
            if (step + 1) % EVAL_STEPS == 0 or step == total_steps - 1:
                # Evaluate
                eval_metrics.reset()
                for _ in range(test_loader.steps_per_epoch):
                    batch = test_loader.next()
                    eval_step(model, eval_metrics, rngs, batch)
                
                eval_results = eval_metrics.compute()
                eval_loss = eval_results['total_loss']
                
                # Print eval results
                print(f"\nProcess {jax.process_index()} - Eval at step {step+1}: " + 
                      " ".join([f"{k}={v:.4f}" for k, v in eval_results.items()]))
                
                # Track best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    print(f"Process {jax.process_index()}: New best eval loss: {best_eval_loss:.4f}")
                
                if jax.process_index() == 0:
                    wandb.log({
                        f"eval/{k}": v for k, v in eval_results.items()
                    } | {
                        'eval/step': step,
                        'eval/epoch': step / steps_per_epoch
                    })
                
                # Save checkpoint with metrics
                metrics = {
                    "train": metrics_values,
                    "eval": eval_results,
                    "best_eval_loss": best_eval_loss
                }
                save_checkpoint(
                    ckpt_manager, 
                    model, 
                    optimizer, 
                    step + 1,
                    train_loader,
                    metrics
                )
                
        progress_bar.close()
        print("Training complete!")

if __name__ == "__main__":
    main()
    