import os
import time
from typing import Tuple

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import resolved dynamically below to avoid import-time execution

CHECKPOINT_PATH = os.path.join("C:", "LLM_Tutorial", "tinygpt_checkpoint.pt")

st.title("TinyGPT — Streamlit UI")
st.markdown("A tiny demo GPT built from transformer blocks. Use this UI to train a tiny model on a toy corpus and generate text.")

# --- Utility to import Block from the project relative path ---
try:
    # Prefer direct import if package path is available
    from transformer_blocks import Block
except Exception:
    # Fallback: load by path using importlib
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "transformer_blocks",
        os.path.join("C:", "LLM_Tutorial", "module_0", "L-2_tinyLLM", "venv", "transformer_blocks.py"),
    )
    transformer_blocks = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = transformer_blocks
    spec.loader.exec_module(transformer_blocks)
    Block = transformer_blocks.Block

# --- Dataset and vocabulary (same small corpus as demo.py) ---
CORPUS = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match",
]
CORPUS = [s + " <END>" for s in CORPUS]
TEXT = " ".join(CORPUS)
WORDS = list(set(TEXT.split()))
VOCAB_SIZE = len(WORDS)
WORD2IDX = {w: i for i, w in enumerate(WORDS)}
IDX2WORD = {i: w for w, i in WORD2IDX.items()}

# Default hyperparams (adapted from demo)
BLOCK_SIZE = 6
EMBED_DIM = 32
N_HEADS = 2
N_LAYERS = 2

# TinyGPT definition (same as demo but kept local so importing demo.py isn't required)
class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, block_size: int = BLOCK_SIZE, embedding_dim: int = EMBED_DIM, n_heads: int = N_HEADS, n_layers: int = N_LAYERS):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens: int = 20):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# Data helpers
DATA = torch.tensor([WORD2IDX[w] for w in TEXT.split()], dtype=torch.long)

@st.cache_data
def get_batch(batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(DATA) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([DATA[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([DATA[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y

# Sidebar controls
st.sidebar.header("Training / Checkpoint")
quick_epochs = st.sidebar.number_input("Quick train epochs", min_value=1, max_value=5000, value=500, step=50)
train_batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=256, value=32)
learning_rate = st.sidebar.number_input("Learning rate", min_value=1e-5, max_value=1.0, value=1e-3, format="%.5f")

col_train, col_ckpt = st.sidebar.columns(2)
train_button = col_train.button("Train (Quick)")
load_button = col_ckpt.button("Load checkpoint")
save_button = st.sidebar.button("Save checkpoint")

# Model state in session
if "model" not in st.session_state:
    st.session_state.model = TinyGPT()
    st.session_state.opt = torch.optim.AdamW(st.session_state.model.parameters(), lr=learning_rate)

model: TinyGPT = st.session_state.model
optimizer = st.session_state.opt

# Load checkpoint
if load_button:
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        st.success(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        st.warning("No checkpoint found. Train the model first.")

# Train action
if train_button:
    epochs = int(quick_epochs)
    batch_size = int(train_batch_size)
    lr = float(learning_rate)
    # Recreate optimizer with new lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    st.session_state.opt = optimizer

    progress_bar = st.progress(0)
    status_text = st.empty()

    model.train()
    for step in range(epochs):
        xb, yb = get_batch(batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, epochs // 20) == 0 or step == epochs - 1:
            status_text.text(f"Step {step+1}/{epochs}, loss={loss.item():.4f}")
        progress_bar.progress((step + 1) / epochs)
    st.success("Training completed")

# Save checkpoint
if save_button:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "opt_state": optimizer.state_dict()}, CHECKPOINT_PATH)
    st.success(f"Saved checkpoint to {CHECKPOINT_PATH}")

# Generation UI
st.header("Generate Text")
seed_text = st.text_input("Seed words (use tokens from the small vocab)", value="hello")
max_tokens = st.slider("Max new tokens", min_value=1, max_value=50, value=15)
generate_button = st.button("Generate")

st.markdown("**Vocabulary (small toy vocab)**")
st.write(", ".join(sorted(WORDS)))

if generate_button:
    # Tokenize seed (split by space and map unknown words to <END> or random)
    tokens = []
    for w in seed_text.split():
        if w in WORD2IDX:
            tokens.append(WORD2IDX[w])
        else:
            # map unknown token to <END> if present else random
            tokens.append(WORD2IDX.get("<END>", int(torch.randint(0, VOCAB_SIZE, (1,)).item())))
    idx = torch.tensor([tokens], dtype=torch.long)
    # Ensure at least one token
    if idx.shape[1] == 0:
        st.warning("Please provide at least one seed word.")
    else:
        model.eval()
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=max_tokens)
        generated = " ".join(IDX2WORD[int(i)] for i in out[0])
        st.markdown("**Generated**")
        st.write(generated)

st.markdown("---")
st.caption("Note: this is a tiny toy model intended for demonstration. Use the Train button to fit it to the tiny corpus before generating meaningful outputs.")
