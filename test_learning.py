import os
import jax
import jax.numpy as jnp
import numpy as np
from datasets import Dataset
from config import *
from flax import nnx
import optax
from train_llm import (
    BatchLoader, create_sharded_model,
    train_step, eval_step, create_learning_rate_schedule
)
from typing import Dict, Tuple, Any

def make_mesh():
    return jax.make_mesh((4, 2), ('expert', 'data'))

def create_challenging_dataset(vocab_size=1000, seq_len=64, num_samples=2000) -> Tuple[Dataset, Dict[str, np.ndarray]]:
    """
    Creates a more challenging dataset with multiple patterns and returns pattern type info.
    """
    input_ids = np.zeros((num_samples, seq_len), dtype=np.int32)
    labels = np.zeros((num_samples, seq_len), dtype=np.int32)
    attention_mask = np.ones((num_samples, seq_len), dtype=np.int32)
    pattern_types = np.zeros(num_samples, dtype=np.int32)  # Track pattern types
    
    for i in range(num_samples):
        pattern_type = i % 4
        pattern_types[i] = pattern_type
        
        if pattern_type == 0:
            # Arithmetic sequence with random start and step
            start = np.random.randint(1, vocab_size // 4)
            step = np.random.randint(1, 10)
            seq = np.array([(start + j * step) % (vocab_size - 1) + 1 for j in range(seq_len)])
            
        elif pattern_type == 1:
            # Modified Fibonacci-like sequence that's easier to learn
            # Start with smaller numbers and use modulo more frequently
            a = np.random.randint(1, 20)  # Smaller initial values
            b = np.random.randint(1, 20)
            seq = np.zeros(seq_len, dtype=np.int32)
            seq[0] = a
            seq[1] = b
            max_val = vocab_size // 4  # Keep values smaller
            for j in range(2, seq_len):
                # Take modulo more frequently to prevent overflow
                next_val = (seq[j-1] + seq[j-2]) % max_val
                # Ensure no zeros in sequence
                seq[j] = next_val + 1 if next_val == 0 else next_val
                
        elif pattern_type == 2:
            # Repeat pattern with random length
            pattern_length = np.random.randint(2, 6)
            pattern = np.random.randint(1, vocab_size // 4, size=pattern_length)  # Keep values smaller
            seq = np.tile(pattern, seq_len // pattern_length + 1)[:seq_len]
            
        else:
            # Base pattern with noise
            start = np.random.randint(1, vocab_size // 4)
            step = np.random.randint(1, 5)
            seq = np.array([(start + j * step) % (vocab_size - 1) + 1 for j in range(seq_len)])
            # Add noise to some positions
            noise_mask = np.random.random(seq_len) < 0.15
            seq[noise_mask] = np.random.randint(1, vocab_size // 4, size=noise_mask.sum())  # Keep noise values smaller
        
        input_ids[i] = seq
        labels[i] = seq
    
    dataset = Dataset.from_dict({
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'pattern_type': pattern_types
    })
    
    # Create pattern-specific masks for evaluation
    pattern_masks = {
        f"pattern_{i}": pattern_types == i for i in range(4)
    }
    
    return dataset, pattern_masks

class PatternSpecificMetric:
    def __init__(self, pattern_masks: Dict[str, np.ndarray]):
        self.pattern_masks = pattern_masks
        self.reset()
    
    def reset(self):
        self.total = {k: 0.0 for k in self.pattern_masks.keys()}
        self.count = {k: 0 for k in self.pattern_masks.keys()}
    
    def update(self, predictions: np.ndarray, labels: np.ndarray, pattern_types: np.ndarray):
        for pattern_name, mask in self.pattern_masks.items():
            pattern_idx = int(pattern_name.split('_')[1])
            pattern_mask = pattern_types == pattern_idx
            if pattern_mask.any():
                correct = (predictions[pattern_mask] == labels[pattern_mask]).mean()
                self.total[pattern_name] += correct
                self.count[pattern_name] += 1
    
    def compute(self):
        return {k: self.total[k] / max(1, self.count[k]) for k in self.pattern_masks.keys()}

def test_model_learning():
    # Configuration for the test
    vocab_size = 1000
    seq_len = 64
    num_samples = 20000
    batch_size = 32
    num_steps = 5000
    
    # Update model config for the test - larger model
    test_config = MODEL_CONFIG.copy()
    test_config.update({
        'vocab_size': vocab_size,
        'd_model': 256,        # 4x larger
        'hidden_dim': 1024,    # 4x larger
        'num_layers': 4,       # 2x more layers
        'num_heads': 8,        # 2x more heads
        'head_dim': 32,        # 2x larger
        'num_experts': 4,
        'num_shared_experts': 0,
    })
    
    # Create challenging dataset with pattern masks
    dataset, pattern_masks = create_challenging_dataset(vocab_size, seq_len, num_samples)
    
    # Setup mesh and data spec
    mesh = make_mesh()
    data_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data', None))
    
    # Create data loader
    data_loader = BatchLoader(dataset=dataset, batch_size=batch_size, data_spec=data_spec)
    
    with mesh:
        # Create model
        model = create_sharded_model()
        
        # Create optimizer with improved learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            total_steps=num_steps,
            warmup_steps=100,    # More warmup steps
            base_lr=3e-4        # Higher learning rate
        )
        
        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    learning_rate=lr_schedule,
                    weight_decay=0.01,
                    b1=0.9,
                    b2=0.999,
                    eps=1e-8
                )
            )
        )
        
        # Create metrics trackers
        metrics = nnx.MultiMetric(
            total_loss=nnx.metrics.Average('total_loss'),
            loss=nnx.metrics.Average('loss'),
            router_loss=nnx.metrics.Average('router_loss'),
            perplexity=nnx.metrics.Average('perplexity'),
            accuracy=nnx.metrics.Average('accuracy')
        )
        
        pattern_metrics = PatternSpecificMetric(pattern_masks)
        
        # Training loop
        print("\nStarting training...")
        print("Step | Loss | Accuracy | Perplexity | Pattern Accuracies")
        print("-" * 80)
        
        initial_metrics = {}
        final_metrics = {}
        best_accuracy = 0.0
        
        for step in range(num_steps):
            batch = data_loader.next()
            loss = train_step(model, optimizer, metrics, batch)
            
            if step == 0:
                initial_metrics = metrics.compute()
                metrics.reset()
            
            if (step + 1) % 50 == 0:  # Print every 50 steps
                current_metrics = metrics.compute()
                
                # Get logits and compute pattern-specific accuracies
                logits, _ = model(batch['input_ids'], batch['attention_mask'])
                predictions = jnp.argmax(logits[..., :-1, :], axis=-1)
                pattern_metrics.update(
                    np.array(predictions),  # Convert JAX array to numpy
                    np.array(batch['labels'][..., 1:]),
                    batch['pattern_type']
                )
                pattern_accs = pattern_metrics.compute()
                pattern_metrics.reset()
                
                # Print metrics
                print(f"{step+1:4d} | {current_metrics['loss']:.4f} | {current_metrics['accuracy']:.4f} | "
                      f"{current_metrics['perplexity']:.4f} | " + 
                      " ".join([f"{k}: {v:.3f}" for k, v in pattern_accs.items()]))
                
                if current_metrics['accuracy'] > best_accuracy:
                    best_accuracy = current_metrics['accuracy']
                    print(f"New best accuracy: {best_accuracy:.4f}")
                
                if step == num_steps - 1:
                    final_metrics = current_metrics
                metrics.reset()
        
        if not initial_metrics or not final_metrics:
            raise RuntimeError("Failed to collect metrics")
        
        print("\nTraining Results:")
        print("-" * 80)
        print(f"Initial Loss: {initial_metrics['loss']:.4f}")
        print(f"Final Loss: {final_metrics['loss']:.4f}")
        print(f"Initial Accuracy: {initial_metrics['accuracy']:.4f}")
        print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"Initial Perplexity: {initial_metrics['perplexity']:.4f}")
        print(f"Final Perplexity: {final_metrics['perplexity']:.4f}")
        print("\nPattern-specific Accuracies:")
        pattern_accs = pattern_metrics.compute()
        for pattern, acc in pattern_accs.items():
            print(f"{pattern}: {acc:.4f}")
        
        # Verify learning occurred with higher standards
        assert final_metrics['loss'] < initial_metrics['loss'], "Model did not learn: loss did not decrease"
        assert final_metrics['accuracy'] > initial_metrics['accuracy'], "Model did not learn: accuracy did not improve"
        assert final_metrics['accuracy'] > 0.8, "Model accuracy below 80%"
        assert final_metrics['perplexity'] < 5.0, "Model perplexity too high"
        
        print("\nTest passed! Model successfully learned the complex patterns.")

if __name__ == "__main__":
    test_model_learning() 