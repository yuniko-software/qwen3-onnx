import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def find_repository_root():
    current = Path.cwd()
    for _ in range(10):
        if (current / "models").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError("Could not locate repository root with 'models' directory")


def get_embedding(model, tokenizer, text):
    batch_dict = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)

    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings[0].cpu().numpy().tolist()


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')

    print("Loading embedding model...")
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
    model.eval()
    print("Model loaded\n")

    test_texts = [
        "What is the capital of France?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "你好，世界！",
        "Привет, мир!",
        "こんにちは世界",
        "مرحبا بالعالم",
        "Hola mundo",
        "Bonjour le monde",
        "Olá mundo",
        "Ciao mondo",
        "Hallo Welt",
        "안녕하세요 세계",
        "<|endoftext|>",
        "<|im_start|>user\nHello<|im_end|>",
        "<|vision_start|><|vision_end|>",
        "<tool_call>function_name</tool_call>",
        "<|fim_prefix|>code before<|fim_suffix|>code after<|fim_middle|>",
        "<|repo_name|>my-repo<|file_sep|>main.py",
        "<think>reasoning process</think>",
        "Empty string test: ",
        "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
        "Numbers: 0123456789",
        "Mixed: Hello世界123!@#"
    ]

    reference_embeddings = {}

    for idx, text in enumerate(test_texts, 1):
        print(f"Generating embedding {idx}/{len(test_texts)}...")
        embedding = get_embedding(model, tokenizer, text)
        reference_embeddings[text] = {
            "embedding": embedding,
            "dimension": len(embedding)
        }

    output_dir = find_repository_root() / "dotnet" / "Qwen3.Onnx.Embedding.Tests" / "TestData"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "reference_embeddings.json"

    print(f"\nSaving reference embeddings to {output_file}")

    with open(output_file, "w") as f:
        json.dump(reference_embeddings, f, indent=2)

    print("Reference embeddings generated successfully!")
    print(f"Total texts: {len(reference_embeddings)}")
    print(f"Embedding dimension: {reference_embeddings[test_texts[0]]['dimension']}")


if __name__ == "__main__":
    main()
