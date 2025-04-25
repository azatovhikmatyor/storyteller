import os
import json

current_dir = os.path.dirname(__file__)

with open(os.path.join(current_dir,"vocab.json"), "r", encoding="utf-8") as f:
    vocab = json.load(f)

itos = {int(k): v for k, v in vocab["itos"].items()}
stoi = vocab["stoi"]

VOCAB_SIZE = len(itos)

end_of_text_token = '<|endoftext|>'
unknown_token = '<|unknown|>'

def encode(text: str):
    encoded = []
    i = 0
    while i < len(text):
        if text.startswith(end_of_text_token, i):
            encoded.append(stoi[end_of_text_token])
            i += len(end_of_text_token)
        else:
            encoded.append(stoi.get(text[i], stoi[unknown_token]))
            i += 1
    return encoded

def decode(tokens: list[int]):
    return ''.join(itos.get(token, unknown_token) for token in tokens)
