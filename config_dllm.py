import jax.numpy as jnp
import os

CONTEXT_LENGTH = 512
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
GRADIENT_CLIP_NORM = 1.0
DTYPE = jnp.bfloat16
PARALLEL_PROCESSING = 8
TOKENIZED_DATASET_PATH = "/mnt/data/tokenized_dataset"
TOKENIZER_NAME = "gpt2"
CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), "checkpoints")
LOG_STEPS = 50
EVAL_STEPS = 5000
SAMPLE_STEPS = 500
MESH_SHAPE = (4, 1)  # (data, expert)
MSE_LOSS_WEIGHT = 0.1

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
    'num_layers': 8,
    'num_heads': 8,
    'head_dim': 64,
    'vocab_size': VOCAB_SIZE,
    'num_experts': 4,
    'num_shared_experts': 1,
    'top_k': 1,
    'capacity_factor': 2.0,
    'min_expert_capacity': 8,
    'max_group_size': 4096,
    'router_z_loss_coef': 1e-3,
    'router_balance_loss_coef': 1e-4,
    'dtype': DTYPE,
    'training': True,
    'use_gradient_checkpointing': True,
}