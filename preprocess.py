from config import *
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dataset(tokenizer):
    dataset = load_dataset(DATASET_CONFIG['path'], name=DATASET_CONFIG['name'], split=DATASET_CONFIG['split'], num_proc=PARALLEL_PROCESSING)

    def tokenize_and_chunk(examples):
        tokenized = tokenizer(examples['text'], truncation=False, add_special_tokens=True)
        chunks = []
        attention_masks = []

        for input_ids in tokenized['input_ids']:
            if len(input_ids) < 32 or not any(input_ids):
                continue

            for i in range(0, len(input_ids), CONTEXT_LENGTH):
                chunk = input_ids[i:i + CONTEXT_LENGTH]
                if len(chunk) < 32:
                    continue

                if len(chunk) < CONTEXT_LENGTH:
                    padding_length = CONTEXT_LENGTH - len(chunk)
                    attention_mask = [1] * len(chunk) + [0] * padding_length
                    chunk = chunk + [tokenizer.eos_token_id] * padding_length
                else:
                    chunk = chunk[:CONTEXT_LENGTH]
                    attention_mask = [1] * CONTEXT_LENGTH

                chunks.append(chunk)
                attention_masks.append(attention_mask)

        return {
            'input_ids': chunks,
            'attention_mask': attention_masks,
            'labels': chunks
        }

    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        remove_columns=dataset.column_names,  # type: ignore
        batched=True,
        num_proc=PARALLEL_PROCESSING,  # type: ignore
        cache_file_name=None  # type: ignore
    )

    tokenized_dataset.save_to_disk(TOKENIZED_DATASET_PATH)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    prepare_dataset(tokenizer)