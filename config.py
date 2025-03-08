import jax.numpy as jnp
import os

CONTEXT_LENGTH = 512
BATCH_SIZE = 128
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
MESH_SHAPE = (4, 4)  # (expert, data)

VOCAB_SIZE = 50257
VOCAB_SIZE = ((VOCAB_SIZE + 127) // 128) * 128

DATASET_CONFIG = {
    'path': 'wikitext',#'HuggingFaceFW/fineweb',
    'name': 'wikitext-103-v1',#'sample-10BT',
    'split': 'train',
}

MODEL_CONFIG = {
    'd_model': 768,
    'hidden_dim': 3072,
    'num_layers': 12,
    'num_heads': 12,
    'head_dim': 64,
    'vocab_size': VOCAB_SIZE,
    'num_experts': 16,
    'num_shared_experts': 2,
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