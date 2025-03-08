import os
import threading
from queue import Queue
import time

from config import *
import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as PS, NamedSharding as NS

from flax import nnx
import optax

from model import Transformer

from tqdm import tqdm
from datasets import load_from_disk
import orbax.checkpoint as ocp
from typing import Dict, Any, Optional, Tuple

def calculate_metrics(logits, labels, mask):
    """Calculate loss, accuracy, and perplexity metrics."""
    shift_logits = logits[..., :-1, :].astype(jnp.float32)
    shift_labels = labels[..., 1:]
    loss_mask = mask[..., :-1]
    
    # Calculate cross entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        shift_logits,
        shift_labels,
    )
    loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)
    
    # Calculate accuracy
    predictions = jnp.argmax(shift_logits, axis=-1)
    correct_predictions = (predictions == shift_labels) * loss_mask
    accuracy = correct_predictions.sum() / (loss_mask.sum() + 1e-9)
    
    # Calculate perplexity
    perplexity = jnp.exp(loss)
    
    return loss, accuracy, perplexity

@nnx.jit
def train_step(model: Transformer, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):

    def loss_fn(model):
        logits, router_loss = model(batch['input_ids'], batch['attention_mask'])
        
        loss, accuracy, perplexity = calculate_metrics(
            logits, batch['labels'], batch['attention_mask']
        )
        
        total_loss = loss + router_loss
        return total_loss, (loss, accuracy, perplexity, router_loss)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (loss, accuracy, perplexity, router_loss)), grads = grad_fn(model)
    
    # Update model parameters
    optimizer.update(grads)
    
    # Update metrics
    metrics.update(
        total_loss=total_loss,
        loss=loss,
        router_loss=router_loss,
        perplexity=perplexity,
        accuracy=accuracy
    )

    return total_loss

@nnx.jit
def eval_step(model: Transformer, metrics: nnx.MultiMetric, batch):

    def loss_fn(model):
        logits, router_loss = model(batch['input_ids'], batch['attention_mask'])
        
        loss, accuracy, perplexity = calculate_metrics(
            logits, batch['labels'], batch['attention_mask']
        )
        
        total_loss = loss + router_loss
        return total_loss, (loss, accuracy, perplexity, router_loss)
    
    total_loss, (loss, accuracy, perplexity, router_loss) = loss_fn(model)
    
    # Update metrics
    metrics.update(
        total_loss=total_loss,
        loss=loss,
        router_loss=router_loss,
        perplexity=perplexity,
        accuracy=accuracy
    )

def make_mesh():
    mesh = jax.make_mesh((1, 2, 2, 2), ("data", "model", "head", "expert"))
    return mesh

def load_dataset():
    # Load the tokenized dataset
    dataset = load_from_disk(TOKENIZED_DATASET_PATH)

    # Take the last batch as test set
    test_size = BATCH_SIZE
    
    # Split the dataset using train_test_split
    split = dataset.train_test_split( # type: ignore
        test_size=test_size,
        shuffle=True,
        seed=42
    )
    
    train_dataset = split['train']
    test_dataset = split['test']
    
    return train_dataset, test_dataset

@nnx.jit
def create_sharded_model():
    model = Transformer(**MODEL_CONFIG, rngs=nnx.Rngs(0))
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

class BatchLoader:
    """Efficient batch loader with parallel prefetching and sharding."""
    def __init__(self, dataset, batch_size: int, data_spec: NS, seed: int = 42, num_prefetch: int = 3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_spec = data_spec
        self.rng = np.random.RandomState(seed)
        self.num_prefetch = num_prefetch
        
        # Thread-safe queue for prefetched batches
        self._prefetch_queue = Queue(maxsize=num_prefetch)
        
        # Calculate number of batches
        self.num_samples = len(dataset)
        self.steps_per_epoch = self.num_samples // batch_size
        
        # Initialize indices with thread lock
        self._lock = threading.Lock()
        self.current_epoch = 0
        self.current_idx = 0
        self.indices = self.rng.permutation(self.num_samples)  # Initialize indices immediately
        
        # Control flags
        self._stop_prefetching = threading.Event()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()
    
    def _shuffle_indices(self):
        """Thread-safe shuffle of dataset indices."""
        with self._lock:
            self.indices = self.rng.permutation(self.num_samples)
            self.current_idx = 0
            self.current_epoch += 1
    
    def _get_next_batch_indices(self):
        """Thread-safe retrieval of next batch indices."""
        with self._lock:
            if self.current_idx + self.batch_size > len(self.indices):
                self._shuffle_indices()
            
            batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
            self.current_idx += self.batch_size
            return batch_indices
    
    def _create_batch(self, indices):
        """Create a sharded batch from indices."""
        examples = {}
        batch_data = self.dataset[indices]
        
        for k, v in batch_data.items():
            array = jnp.array(v)
            examples[k] = jax.device_put(array, self.data_spec)
        
        return examples
    
    def _prefetch_worker(self):
        """Background worker that continuously prefetches batches."""
        while not self._stop_prefetching.is_set():
            try:
                # Get next batch indices
                batch_indices = self._get_next_batch_indices()
                
                # Create batch
                batch = self._create_batch(batch_indices)
                
                # Add to queue with timeout to allow checking stop flag
                self._prefetch_queue.put(batch, timeout=1.0)
            except:
                # If queue is full or other error, sleep briefly
                time.sleep(0.1)
    
    def next(self):
        """Get next batch from the prefetch queue."""
        try:
            return self._prefetch_queue.get(timeout=10.0)
        except:
            raise RuntimeError("Timeout waiting for next batch. Prefetching thread may have died.")
    
    def __del__(self):
        """Cleanup method to stop background thread."""
        self._stop_prefetching.set()
        if hasattr(self, '_prefetch_thread'):
            self._prefetch_thread.join(timeout=5.0)

def count_params(model: Transformer) -> float:
    """Count total number of trainable parameters in billions."""
    total = 0
    state = nnx.state(model)
    for param in jax.tree_util.tree_leaves(state):
        if isinstance(param, jnp.ndarray):
            total += param.size
    return total / 1e9  # Convert to billions

def create_checkpoint_manager(base_dir: str = "checkpoints", max_to_keep: int = 5) -> ocp.CheckpointManager:
    """Create an Orbax checkpoint manager."""
    os.makedirs(base_dir, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=True,
    )
    return ocp.CheckpointManager(
        base_dir,
        checkpointer,
        options=options,
    )

def save_checkpoint(
    ckpt_manager: ocp.CheckpointManager, 
    model: Transformer, 
    optimizer: nnx.Optimizer, 
    step: int,
    batch_loader: BatchLoader,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Save the model, optimizer state, current step, and batch state to checkpoint."""
    # Get states to save
    model_state = nnx.state(model)
    optimizer_state = nnx.state(optimizer)
    
    # Create batch state dict to save batch loader state
    batch_state = {
        "current_epoch": batch_loader.current_epoch,
        "current_idx": batch_loader.current_idx,
        "indices": np.array(batch_loader.indices),  # Convert to numpy for serialization
    }
    
    # Combine everything into one state dict
    ckpt_state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "step": step,
        "batch_state": batch_state,
        "metrics": metrics or {},
    }
    
    # Save the checkpoint
    ckpt_manager.save(step, ckpt_state)
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(
    ckpt_manager: ocp.CheckpointManager,
    model: Transformer,
    optimizer: nnx.Optimizer,
    batch_loader: Optional[BatchLoader] = None
) -> Tuple[int, Dict[str, Any]]:
    """Load checkpoint into existing model and optimizer if available."""
    # Check if checkpoint exists
    step = ckpt_manager.latest_step()
    if step is None:
        print("No checkpoint found, starting from scratch")
        return 0, {}
    
    # Create abstract target based on the existing model and optimizer
    abs_model_state = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if hasattr(x, 'shape') else x,
        nnx.state(model)
    )
    
    abs_optimizer_state = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if hasattr(x, 'shape') else x,
        nnx.state(optimizer)
    )
    
    # Create target dict for restoration
    abs_target = {
        "model": abs_model_state,
        "optimizer": abs_optimizer_state,
        "step": 0,
        "batch_state": {
            "current_epoch": 0,
            "current_idx": 0, 
            "indices": np.array([]),
        },
        "metrics": {},
    }
    
    # Restore checkpoint
    print(f"Restoring checkpoint from step {step}")
    restored = ckpt_manager.restore(step, abs_target)
    
    # Update model and optimizer states
    nnx.state(model).update(restored["model"])
    nnx.state(optimizer).update(restored["optimizer"])
    
    # Restore batch loader state if provided
    if batch_loader is not None:
        batch_state = restored["batch_state"]
        with batch_loader._lock:
            batch_loader.current_epoch = batch_state["current_epoch"]
            batch_loader.current_idx = batch_state["current_idx"]
            # Convert to numpy array first to ensure compatibility
            batch_loader.indices = np.array(batch_state["indices"])
    
    return restored["step"], restored.get("metrics", {})

def main():
    train_dataset, test_dataset = load_dataset()

    mesh = make_mesh()
    data_spec = NS(mesh, PS('data', None))

    # Create batch loaders
    train_loader = BatchLoader(dataset=train_dataset, batch_size=BATCH_SIZE, data_spec=data_spec)
    test_loader = BatchLoader(dataset=test_dataset, batch_size=BATCH_SIZE, data_spec=data_spec, seed=43)

    # Create checkpoint manager
    ckpt_manager = create_checkpoint_manager(CHECKPOINT_DIR)
    
    # Calculate training steps
    num_epochs = NUM_EPOCHS
    steps_per_epoch = train_loader.steps_per_epoch
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = min(2000, total_steps // 10)  # 10% of total steps or 2000, whichever is smaller

    with mesh:
        # Create model
        model = create_sharded_model()
        
        # Print model size
        print(f"\nModel Parameters: {count_params(model):.2f}B")
        print("-" * 50)

        # Create learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=LEARNING_RATE
        )

        # Create optimizer
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
        
        # Try to restore from checkpoint
        start_step, metrics = load_checkpoint(ckpt_manager, model, optimizer, train_loader)
        
        # Initialize best eval loss from checkpoint metrics or default
        best_eval_loss = metrics.get("best_eval_loss", float('inf'))
        if start_step > 0:
            print(f"Resuming from step {start_step} with best eval loss: {best_eval_loss:.4f}")
        
        # Create metrics tracker
        train_metrics = nnx.MultiMetric(
            total_loss=nnx.metrics.Average('total_loss'),
            loss=nnx.metrics.Average('loss'),
            router_loss=nnx.metrics.Average('router_loss'),
            perplexity=nnx.metrics.Average('perplexity'),
            accuracy=nnx.metrics.Average('accuracy')
        )
        
        eval_metrics = nnx.MultiMetric(
            total_loss=nnx.metrics.Average('total_loss'),
            loss=nnx.metrics.Average('loss'),
            router_loss=nnx.metrics.Average('router_loss'),
            perplexity=nnx.metrics.Average('perplexity'),
            accuracy=nnx.metrics.Average('accuracy')
        )

        # Main training loop
        progress_bar = tqdm(total=total_steps - start_step, desc=f"Training")
        for step in range(start_step, total_steps):
            batch = train_loader.next()
            train_step(model, optimizer, train_metrics, batch)
            
            if (step + 1) % LOG_STEPS == 0 or step == total_steps - 1:
                metrics_values = train_metrics.compute()
                train_metrics.reset()
                
                progress_bar.set_postfix({
                        'loss': f"{metrics_values['loss']:.4f}",
                        'ppl': f"{metrics_values['perplexity']:.4f}",
                        'lr': f"{lr_schedule(step):.6f}"
                })
                progress_bar.update(LOG_STEPS)
            
            # Evaluate and save checkpoint periodically
            if (step + 1) % EVAL_STEPS == 0 or step == total_steps - 1:
                # Evaluate
                eval_metrics.reset()
                for _ in range(test_loader.steps_per_epoch):
                    batch = test_loader.next()
                    eval_step(model, eval_metrics, batch)
                
                eval_results = eval_metrics.compute()
                eval_loss = eval_results['loss']
                
                # Print eval results
                print(f"\nEval at step {step+1}: " + 
                      " ".join([f"{k}={v:.4f}" for k, v in eval_results.items()]))
                
                # Track best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    print(f"New best eval loss: {best_eval_loss:.4f}")
                
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
                    step + 1,  # Save as the next step
                    train_loader,
                    metrics
                )
                
        progress_bar.close()
        print("Training complete!")

if __name__ == "__main__":
    main()