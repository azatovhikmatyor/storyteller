import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import tokenizer.tokenizer as tokenizer
from dataset import get_data_loader
from models import GPT, GPTConfig


@torch.no_grad()
def evaluate(model: Module, dl: DataLoader, device='cpu'):
    model = model.to(device)
    model.eval()

    lossi = []
    for X, Y in tqdm(dl, desc='Evaluating...'):
        X, Y = X.to(device), Y.to(device)
        # Y = Y[:, -1]
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), Y.view(-1))
        lossi.append(loss)
    
    return torch.tensor(lossi).mean().item()


def train_one_epoch(
        model: Module, 
        optimizer: Optimizer, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        device='cpu',
        return_tr_loss=True,
        return_val_loss=True):
    model.train()

    lossi = []

    for X, Y in tqdm(train_loader, desc='Training...'):
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossi.append(loss.item())
    
    tr_loss = None
    if return_tr_loss is True:
        tr_loss = evaluate(model, train_loader, device)

    val_loss = None   
    if return_val_loss is True:
        val_loss = evaluate(model, val_loader, device=device)

    return {
        'tr_loss': tr_loss,
        'val_loss': val_loss,
        'training_losses': lossi
    }



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = tokenizer.VOCAB_SIZE

    # GPT Hyperparameters
    block_size = 32
    n_embd = 32
    num_attn_heads = 8
    num_decoder_layers = 6
    drop_rate = 0.1

    tr_loader = get_data_loader(split='train', batch_size=16, block_size=block_size, stride=4, shuffle=True)
    val_loader = get_data_loader(split='validation', batch_size=16, block_size=block_size, stride=4, shuffle=False)


    config = GPTConfig(vocab_size, n_embd, block_size, num_attn_heads,num_decoder_layers)
    model = GPT(config)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    out = model.generate()
    print(out['out'])



    lossi = []
    epoch_train_lossi = []
    epoch_val_lossi = []

    best_loss = torch.inf

    val_epoch_tol = 5
    not_val_decreased = 0

    epoch = 1

    while True:
        if epoch == 45:
            break

        model.train()
        print(epoch)
        for X, Y in tr_loader:
            logits = model(X)

            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), Y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossi.append(loss.item())
        
        tr_loss = evaluate(model, val_loader, device)
        val_loss = evaluate(model, tr_loader, device)
        epoch_train_lossi.append(tr_loss)
        epoch_val_lossi.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'gpt_checkpoint.pth')
            not_val_decreased = 0
        else:
            not_val_decreased += 1

        if not_val_decreased == val_epoch_tol:
            print(f"Validation loss is not decreasing since last {not_val_decreased} epochs.")
            break

        epoch += 1


    plt.plot(lossi)
    plt.plot(epoch_train_lossi, label='train loss')
    plt.plot(epoch_val_lossi, label='validation loss')
    plt.legend()
    plt.show()

    checkpoint = torch.load('gpt_checkpoint.pth', weights_only=False)
    checkpoint['epoch']

    config = GPTConfig(vocab_size, n_embd, block_size, num_attn_heads,num_decoder_layers)
    best_model = GPT(config)
    best_model.load_state_dict(checkpoint['model_state_dict'])

    @torch.no_grad()
    def generate(model:nn.Module, start_token:int=0, max_length:int=5000):
        model.eval()
        out = []
        context = [start_token]
        for i in range(max_length):
            logits = model(torch.tensor([context]))[:, -1, :]
            # print(logits.shape)
            prob = F.softmax(logits, dim=-1)
            # print(prob.shape)
            ix = torch.multinomial(prob, num_samples=1).item()
            
            out.append(ix)
            if ix == 0:
                break
            context = context[-block_size+1:] + [ix]
        return out

    out = generate(best_model)
    print(tokenizer.decode(out))
