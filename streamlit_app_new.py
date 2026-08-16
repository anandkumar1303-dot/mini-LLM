import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import streamlit as st
else:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        class _MissingStreamlit:
            def __getattr__(self, _name):
                raise ModuleNotFoundError(
                    "streamlit is not installed. Install it with: pip install streamlit"
                )

        st = _MissingStreamlit()

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import sys

# Resolve transformer_blocks (load from file path so the app is independent of package layout)
TB_PATH = os.path.join("C:\\", "LLM_Tutorial", "module_0", "L-2_tinyLLM", "venv", "transformer_blocks.py")
if not os.path.exists(TB_PATH):
    raise FileNotFoundError(f"transformer_blocks.py not found at {TB_PATH}")
spec = importlib.util.spec_from_file_location("transformer_blocks", TB_PATH)
transformer_blocks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_blocks)
Block = transformer_blocks.Block

# Toy corpus (same as demo)
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

# Hyperparams
BLOCK_SIZE = 6
EMBED_DIM = 32
N_HEADS = 2
N_LAYERS = 2

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

# Prepare data
DATA = torch.tensor([WORD2IDX[w] for w in TEXT.split()], dtype=torch.long)

@st.cache_data
def get_batch(batch_size: int = 16):
    ix = torch.randint(len(DATA) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([DATA[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([DATA[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y

# Streamlit UI
st.title("TinyGPT — Streamlit UI")
st.write("A tiny demo GPT using local transformer blocks.")

# Sidebar controls
st.sidebar.header("Training")
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=5000, value=300)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=256, value=32)
lr = st.sidebar.number_input("Learning rate", min_value=1e-5, max_value=1.0, value=1e-3, format="%.5f")

# Model in session state
if "model" not in st.session_state:
    st.session_state.model = TinyGPT()
    st.session_state.opt = torch.optim.AdamW(st.session_state.model.parameters(), lr=lr)

model: TinyGPT = st.session_state.model
optimizer = st.session_state.opt

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Train"):
        model.train()
        progress = st.progress(0)
        status = st.empty()
        for e in range(int(epochs)):
            xb, yb = get_batch(int(batch_size))
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if e % max(1, int(epochs)//10) == 0:
                status.text(f"Epoch {e+1}/{epochs} loss={loss.item():.4f}")
            progress.progress((e+1)/int(epochs))
        st.success("Training finished")

with col2:
    if st.button("Save checkpoint"):
        ckpt_path = os.path.join("C:\\", "LLM_Tutorial", "tinygpt_checkpoint.pt")
        torch.save({"model_state": model.state_dict(), "opt_state": optimizer.state_dict()}, ckpt_path)
        st.write(f"Saved checkpoint to {ckpt_path}")

with col3:
    if st.button("Load checkpoint"):
        ckpt_path = os.path.join("C:\\", "LLM_Tutorial", "tinygpt_checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["opt_state"])
            st.write("Checkpoint loaded")
        else:
            st.warning("No checkpoint found")

st.subheader("Generate")
seed = st.text_input("Seed words (space-separated)", value="hello")
max_new = st.slider("Max new tokens", 1, 50, 15)

if st.button("Generate"):
    tokens = [WORD2IDX.get(w, WORD2IDX.get("<END>", 0)) for w in seed.split()]
    idx = torch.tensor([tokens], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=int(max_new))
    generated = " ".join(IDX2WORD[int(i)] for i in out[0])
    st.write(generated)

st.markdown("**Vocab:**")
st.write(", ".join(sorted(WORDS)))
