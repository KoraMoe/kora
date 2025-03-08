import jax.numpy as jnp
import os

CONTEXT_LENGTH = 512
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
GRADIENT_CLIP_NORM = 1.0
DTYPE = jnp.bfloat16
PARALLEL_PROCESSING = 8
TOKENIZED_DATASET_PATH = "/mnt/data/tokenized_dataset"
TOKENIZER_NAME = "gpt2"
CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), "checkpoints")
LOG_STEPS = 100
EVAL_STEPS = 1000
MESH_SHAPE = (4, 2, 2, 2) # (data, model, head, expert)

VOCAB_SIZE = 50257
VOCAB_SIZE = ((VOCAB_SIZE + 127) // 128) * 128

DATASET_CONFIG = {
    'path': 'wikitext',#'HuggingFaceFW/fineweb',
    'name': 'wikitext-103-v1',#'sample-10BT',
    'split': 'train',
}

MODEL_CONFIG = {
    'd_model': 512,
    'hidden_dim': 2048,
    'num_layers': 12,
    'num_heads': 8,
    'head_dim': 64,
    'vocab_size': VOCAB_SIZE,
    'num_experts': 8,
    'num_shared_experts': 2,
    'top_k': 2,
    'capacity_factor': 1.5,
    'min_expert_capacity': 8,
    'max_group_size': 4096,
    'router_z_loss_coef': 1e-3,
    'router_balance_loss_coef': 1e-4,
    'dtype': DTYPE,
    'training': True,
    'use_gradient_checkpointing': True,
}