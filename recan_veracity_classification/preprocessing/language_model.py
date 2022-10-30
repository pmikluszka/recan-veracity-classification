import gc
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Tuple

tokenizer = None
model = None

# name of the LM from HuggingFace
MODEL_NAME = os.environ.get("LANGUAGE_MODEL", "cardiffnlp/twitter-roberta-base")
DEVICE = (
    f"cuda:{torch.cuda.current_device()}"
    if torch.cuda.is_available()
    else "cpu"
)


def _preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def _assign_device(tokenizer_output: dict) -> dict:
    tokens_tensor = tokenizer_output["input_ids"].to(DEVICE)
    attention_mask = tokenizer_output["attention_mask"].to(DEVICE)

    output = {
        "input_ids": tokens_tensor,
        "attention_mask": attention_mask,
    }

    return output


def get_embedding(
    text: str, max_words: int = 30
) -> Tuple[torch.Tensor, torch.Tensor]:
    global tokenizer
    global model
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if model == None:
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    text = _preprocess(text)
    encoded_input = _assign_device(tokenizer(text, return_tensors="pt"))
    features = model(**encoded_input)[0].detach().cpu()[0]

    words = features.shape[0]
    mask = torch.zeros(max_words, dtype=torch.bool)
    mask[words:] = True

    features = F.pad(features, (0, 0, 0, max_words - words))

    return features, mask


def cleanup_lm():
    global tokenizer
    global model
    del tokenizer
    del model
    gc.collect()
