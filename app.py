# import streamlit as st
# import pandas as pd
# import torch
# import torch.nn as nn
# import re
# import math
# import pickle

# # --- Model Definition ---
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=50):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1), :]

# class TransformerSeq2Seq(nn.Module):
#     def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, num_encoder_layers=3, 
#                  num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
#         super(TransformerSeq2Seq, self).__init__()
#         self.src_embedding = nn.Embedding(src_vocab_size, d_model)
#         self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.transformer = nn.Transformer(
#             d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout
#         )
#         self.fc_out = nn.Linear(d_model, tgt_vocab_size)
#         self.d_model = d_model

#     def generate_square_subsequent_mask(self, sz):
#         mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
#         return mask

#     def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
#         src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
#         tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
#         src_emb = self.pos_encoder(src_emb)
#         tgt_emb = self.pos_encoder(tgt_emb)
#         src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, d_model)
#         tgt_emb = tgt_emb.permute(1, 0, 2)  # (seq_len, batch, d_model)
#         # Debug shapes before Transformer
#         print(f"src_emb shape: {src_emb.shape}, tgt_emb shape: {tgt_emb.shape}")
#         if src_padding_mask is not None:
#             print(f"src_padding_mask shape in forward: {src_padding_mask.shape}")
#         if tgt_padding_mask is not None:
#             print(f"tgt_padding_mask shape in forward: {tgt_padding_mask.shape}")
#         output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, src_key_padding_mask=src_padding_mask, 
#                                  tgt_key_padding_mask=tgt_padding_mask)
#         output = output.permute(1, 0, 2)  # (batch, seq_len, d_model)
#         return self.fc_out(output)

# # --- Tokenization and Sequence Functions ---
# def tokenize(text):
#     if not isinstance(text, str):
#         text = '' if pd.isna(text) else str(text)
#     text = re.sub(r'([(),=+\-*/%;{}<>!?:])', r' \1 ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text.split()

# MAX_LEN = 50
# def to_sequence(tokens, vocab, max_len=MAX_LEN):
#     seq = [vocab['<START>']] + [vocab.get(token, vocab['<UNK>']) for token in tokens[:max_len-2]] + [vocab['<END>']]
#     padded_seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
#     assert len(padded_seq) == max_len, f"Expected sequence length {max_len}, but got {len(padded_seq)}"
#     return padded_seq

# def inference(model, src_seq, vocab_src, vocab_tgt, max_len=MAX_LEN):
#     model.eval()
    
#     # Debug: Check the length of src_seq
#     print(f"src_seq length: {len(src_seq)}")
#     assert len(src_seq) == max_len, f"src_seq length should be {max_len}, but got {len(src_seq)}"

#     # Convert to tensor
#     src = torch.tensor([src_seq], dtype=torch.long).to(device)
#     print(f"src shape: {src.shape}")  # Should be (1, 50)

#     # Create padding mask for src
#     src_padding_mask = (src == vocab_src['<PAD>']).to(device)
#     print(f"src_padding_mask shape: {src_padding_mask.shape}")  # Should be (1, 50)
#     assert src_padding_mask.shape == (1, max_len), f"Expected src_padding_mask shape (1, {max_len}), but got {src_padding_mask.shape}"

#     tgt = torch.tensor([[vocab_tgt['<START>']]], dtype=torch.long).to(device)

#     with torch.no_grad():
#         for i in range(max_len):
#             tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
#             tgt_padding_mask = (tgt == vocab_tgt['<PAD>']).to(device)
#             print(f"Iteration {i}: tgt shape: {tgt.shape}, tgt_padding_mask shape: {tgt_padding_mask.shape}")
#             output = model(src, tgt, src_mask=None, tgt_mask=tgt_mask, 
#                          src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
#             next_token = output[:, -1, :].argmax(dim=-1).item()
#             if next_token == vocab_tgt['<END>']:
#                 break
#             tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
    
#     return [list(vocab_tgt.keys())[list(vocab_tgt.values()).index(t)] for t in tgt[0].tolist()[1:]]

# # --- Load Vocabularies and Models ---
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open('pseudo_vocab.pkl', 'rb') as f:
#     pseudo_vocab = pickle.load(f)
# with open('cpp_vocab.pkl', 'rb') as f:
#     cpp_vocab = pickle.load(f)

# model_q1 = TransformerSeq2Seq(len(pseudo_vocab), len(cpp_vocab)).to(device)
# model_q2 = TransformerSeq2Seq(len(cpp_vocab), len(pseudo_vocab)).to(device)

# model_q1.load_state_dict(torch.load('model_q1.pth', map_location=device))
# model_q2.load_state_dict(torch.load('model_q2.pth', map_location=device))

# # --- Streamlit App ---
# st.title("Code Translator with Transformer")

# tab1, tab2 = st.tabs(["Pseudocode to C++", "C++ to Pseudocode"])

# with tab1:
#     st.header("Pseudocode to C++")
#     pseudo_input = st.text_area("Enter Pseudocode", placeholder="e.g., if b = 1 return a , else call function gcd ( b , a % b )")
#     if st.button("Translate to C++"):
#         if pseudo_input:
#             tokens = tokenize(pseudo_input)
#             print(f"Tokens: {tokens}")
#             pseudo_seq = to_sequence(tokens, pseudo_vocab)
#             print(f"Pseudo sequence: {pseudo_seq}")
#             cpp_output = inference(model_q1, pseudo_seq, pseudo_vocab, cpp_vocab)
#             cpp_cleaned = "".join([token if token in "(),;{}" else " " + token for token in cpp_output]).strip()
#             st.code(cpp_cleaned, language="cpp")
#         else:
#             st.warning("Please enter some pseudocode.")

# with tab2:
#     st.header("C++ to Pseudocode")
#     cpp_input = st.text_area("Enter C++ Code", placeholder="e.g., return ! b ? a : gcd ( b , a % b ) ;")
#     if st.button("Translate to Pseudocode"):
#         if cpp_input:
#             tokens = tokenize(cpp_input)
#             print(f"Tokens: {tokens}")
#             cpp_seq = to_sequence(tokens, cpp_vocab)
#             print(f"CPP sequence: {cpp_seq}")
#             pseudo_output = inference(model_q2, cpp_seq, cpp_vocab, pseudo_vocab)
#             pseudo_cleaned = " ".join(pseudo_output)
#             st.write(pseudo_cleaned)
#         else:
#             st.warning("Please enter some C++ code.")

# st.markdown("---")
# st.write("Built with Streamlit and PyTorch. Trained on SPOC dataset.")


import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import re
import math
import pickle
import gdown
import os

# --- Model Definition ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, num_encoder_layers=3, 
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerSeq2Seq, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, d_model)
        tgt_emb = tgt_emb.permute(1, 0, 2)  # (seq_len, batch, d_model)
        print(f"src_emb shape: {src_emb.shape}, tgt_emb shape: {tgt_emb.shape}")
        if src_padding_mask is not None:
            print(f"src_padding_mask shape in forward: {src_padding_mask.shape}")
        if tgt_padding_mask is not None:
            print(f"tgt_padding_mask shape in forward: {tgt_padding_mask.shape}")
        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, src_key_padding_mask=src_padding_mask, 
                                 tgt_key_padding_mask=tgt_padding_mask)
        output = output.permute(1, 0, 2)  # (batch, seq_len, d_model)
        return self.fc_out(output)

# --- Tokenization and Sequence Functions ---
def tokenize(text):
    if not isinstance(text, str):
        text = '' if pd.isna(text) else str(text)
    text = re.sub(r'([(),=+\-*/%;{}<>!?:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

MAX_LEN = 50
def to_sequence(tokens, vocab, max_len=MAX_LEN):
    seq = [vocab['<START>']] + [vocab.get(token, vocab['<UNK>']) for token in tokens[:max_len-2]] + [vocab['<END>']]
    padded_seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
    assert len(padded_seq) == max_len, f"Expected sequence length {max_len}, but got {len(padded_seq)}"
    return padded_seq

def inference(model, src_seq, vocab_src, vocab_tgt, max_len=MAX_LEN):
    model.eval()
    print(f"src_seq length: {len(src_seq)}")
    assert len(src_seq) == max_len, f"src_seq length should be {max_len}, but got {len(src_seq)}"

    src = torch.tensor([src_seq], dtype=torch.long).to(device)
    print(f"src shape: {src.shape}")

    src_padding_mask = (src == vocab_src['<PAD>']).to(device)
    print(f"src_padding_mask shape: {src_padding_mask.shape}")
    assert src_padding_mask.shape == (1, max_len), f"Expected src_padding_mask shape (1, {max_len}), but got {src_padding_mask.shape}"

    tgt = torch.tensor([[vocab_tgt['<START>']]], dtype=torch.long).to(device)

    with torch.no_grad():
        for i in range(max_len):
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            tgt_padding_mask = (tgt == vocab_tgt['<PAD>']).to(device)
            print(f"Iteration {i}: tgt shape: {tgt.shape}, tgt_padding_mask shape: {tgt_padding_mask.shape}")
            output = model(src, tgt, src_mask=None, tgt_mask=tgt_mask, 
                         src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            if next_token == vocab_tgt['<END>']:
                break
            tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
    
    return [list(vocab_tgt.keys())[list(vocab_tgt.values()).index(t)] for t in tgt[0].tolist()[1:]]

# --- Download Models from Google Drive ---
def download_model_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(output_path):
        print(f"Downloading {output_path} from Google Drive...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists, skipping download.")

# Model file IDs from Google Drive
model_q1_id = "1E6Ro_BqMwLzw4JYchOFNZ4dfeQ-dYTXm"
model_q2_id = "14cNGxgkjRIaP_6Mtp1xB5wD2NyWCHZS4"

# Download models
download_model_from_drive(model_q1_id, "model_q1.pth")
download_model_from_drive(model_q2_id, "model_q2.pth")

# --- Load Vocabularies and Models ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('pseudo_vocab.pkl', 'rb') as f:
    pseudo_vocab = pickle.load(f)
with open('cpp_vocab.pkl', 'rb') as f:
    cpp_vocab = pickle.load(f)

model_q1 = TransformerSeq2Seq(len(pseudo_vocab), len(cpp_vocab)).to(device)
model_q2 = TransformerSeq2Seq(len(cpp_vocab), len(pseudo_vocab)).to(device)

model_q1.load_state_dict(torch.load('model_q1.pth', map_location=device))
model_q2.load_state_dict(torch.load('model_q2.pth', map_location=device))

# --- Streamlit App ---
st.title("Code Translator with Transformer 🎉")

st.write("Welcome to the Code Translator! 🌟 Convert pseudocode to C++ or C++ to pseudocode using a Transformer model trained on the SPOC dataset. 🚀")
st.write(f"Check out my GitHub profile: [Seemal Zia](https://github.com/seemalch) 👨‍💻")

tab1, tab2 = st.tabs(["Pseudocode to C++ 🌈", "C++ to Pseudocode 🌌"])

with tab1:
    st.header("Pseudocode to C++ ✨")
    pseudo_input = st.text_area("Enter Pseudocode", placeholder="e.g., if b = 1 return a , else call function gcd ( b , a % b )")
    if st.button("Translate to C++ 🚀"):
        if pseudo_input:
            tokens = tokenize(pseudo_input)
            print(f"Tokens: {tokens}")
            pseudo_seq = to_sequence(tokens, pseudo_vocab)
            print(f"Pseudo sequence: {pseudo_seq}")
            cpp_output = inference(model_q1, pseudo_seq, pseudo_vocab, cpp_vocab)
            cpp_cleaned = "".join([token if token in "(),;{}" else " " + token for token in cpp_output]).strip()
            st.code(cpp_cleaned, language="cpp")
        else:
            st.warning("Please enter some pseudocode! 😄")

with tab2:
    st.header("C++ to Pseudocode 🌠")
    cpp_input = st.text_area("Enter C++ Code", placeholder="e.g., return ! b ? a : gcd ( b , a % b ) ;")
    if st.button("Translate to Pseudocode 🚀"):
        if cpp_input:
            tokens = tokenize(cpp_input)
            print(f"Tokens: {tokens}")
            cpp_seq = to_sequence(tokens, cpp_vocab)
            print(f"CPP sequence: {cpp_seq}")
            pseudo_output = inference(model_q2, cpp_seq, cpp_vocab, pseudo_vocab)
            pseudo_cleaned = " ".join(pseudo_output)
            st.write(pseudo_cleaned)
        else:
            st.warning("Please enter some C++ code! 😄")

st.markdown("---")
st.write("Built with Streamlit and PyTorch. Trained on SPOC dataset. ❤️")
