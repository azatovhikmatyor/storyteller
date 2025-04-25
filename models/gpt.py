import os
import json
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import tokenizer


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    n_embd: int
    block_size: int
    num_attn_heads: int
    num_decoder_layers: int = 6
    drop_rate: int = 0.1


class DecoderBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.layer_norm1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd,
                                          num_heads=config.num_attn_heads,
                                          dropout=config.drop_rate,
                                          batch_first=True)
        self.layer_norm2 = nn.LayerNorm(config.n_embd)
        self.ffnn = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd*4),
            nn.GELU(),
            nn.Linear(config.n_embd*4, config.n_embd)
        )
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.shape
        identity = x
        x = self.layer_norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=self.mask[:T, :T])
        x = x + identity
        
        identity = x
        x = self.layer_norm2(x)
        x = self.ffnn(x)
        x = x + identity
        return x
    


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop_emb = nn.Dropout(config.drop_rate)
        
        self.decoder = nn.Sequential(
            *[DecoderBlock(config) for _ in range(config.num_decoder_layers)]
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.decoder(x)
        x = self.lm_head(x)
        return x   

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        with open(os.path.join(pretrained_model_name_or_path, 'config.json'), 'rt', encoding='utf-8') as f:
            config_json = json.load(f)

        config = GPTConfig(**config_json)
        model = GPT(config)
        model.load_state_dict(torch.load(os.path.join(pretrained_model_name_or_path, 'checkpoints.pth'), weights_only=True))
        return model
    
    def save_checkpoints(self, model_name_or_path: str):
        if not os.path.exists(model_name_or_path):
            os.makedirs(model_name_or_path)

        torch.save(self.state_dict(), os.path.join(model_name_or_path, 'checkpoints.pth'))
        with open(os.path.join(model_name_or_path, 'config.json'), 'wt', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=4)
        
        print(f"Model checkpoints saved to {model_name_or_path!r}")

    
    @torch.no_grad()
    def generate(
        self,
        context: str = None,
        max_tokens:int = 100, 
        stream:bool = False, 
        device='cpu', eos_token='<|endoftext|>'):
        
        self.to(device)
        self.eval()

        eos_token_ix = tokenizer.encode(eos_token)[0]
        max_context_length = self.config.block_size
        context = context if context is not None else eos_token
        
        encoded = tokenizer.encode(context)
        encoded = torch.tensor(encoded).unsqueeze(0)

        out = []

        for _ in range(max_tokens):
            logits = self(encoded)[0,-1,:]
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            if ix.item() == eos_token_ix:
                break

            encoded = torch.cat([encoded, ix.unsqueeze(0)], dim=-1)[:, -max_context_length:]
            out.append(ix.item())

        out = tokenizer.decode(out)
        return {
            'context': context,
            'out': out
        }



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = tokenizer.VOCAB_SIZE
    block_size = 16
    n_embd = 32
    num_attn_heads = 8
    num_decoder_layers = 6
    drop_rate = 0.1


    config = GPTConfig(vocab_size, n_embd, block_size, num_attn_heads,num_decoder_layers)
    model = GPT(config)

    
