"""
Urdu Conversational Chatbot - Streamlit Interface
Transformer Encoder-Decoder Architecture
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
import json
import re
from pathlib import Path
from typing import List, Tuple
import unicodedata

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="اردو چیٹ بوٹ | Urdu Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR RTL AND STYLING
# ============================================================
st.markdown("""
<style>
    /* Import Urdu fonts */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    /* RTL Support */
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Alvi Nastaleeq', serif;
        line-height: 2.0;
        unicode-bidi: plaintext;
    }
    
    /* Chat Messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0;
        max-width: 75%;
        float: right;
        clear: both;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 0;
        max-width: 75%;
        float: left;
        clear: both;
        box-shadow: 0 2px 8px rgba(245, 87, 108, 0.3);
    }
    
    .message-label {
        font-size: 11px;
        opacity: 0.8;
        margin-bottom: 5px;
        font-weight: bold;
    }
    
    .message-text {
        font-size: 16px;
        line-height: 1.8;
    }
    
    .chat-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        min-height: 150px;
        max-height: 600px;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Input styling */
    .stTextArea textarea {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        text-align: right;
        font-size: 16px;
        line-height: 1.8;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TEXT NORMALIZATION
# ============================================================
ZW_RE = re.compile(r"[\u200c\u200d]")
TAT_RE = re.compile(r"\u0640")
DIAC_RE = re.compile(r"[\u064B-\u0652\u0670\u0653-\u065F\u06D6-\u06ED]")

LETTER_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ي": "ی", "ى": "ی", "ئ": "ی", "یٰ": "ی",
    "ة": "ہ", "ھ": "ہ", "ۀ": "ہ", "ھ": "ہ", "ہٰ": "ہ",
    "ك": "ک",
    "ؤ": "و",
}

PUNCT_MAP = {
    "،": ",", "؛": ";", "۔": ".", "؟": "?",
    "٬": ",", "٫": ".",
    "«": '"', "»": '"', "‹": '"', "›": '"',
}

E2W = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def normalize_urdu(text: str) -> str:
    """Normalize Urdu text for consistent processing"""
    if not isinstance(text, str):
        return ""
    
    s = unicodedata.normalize("NFKC", text)
    s = ZW_RE.sub("", s)
    s = TAT_RE.sub("", s)
    s = DIAC_RE.sub("", s)
    
    for src, dst in LETTER_MAP.items():
        s = s.replace(src, dst)
    
    for k, v in PUNCT_MAP.items():
        s = s.replace(k, v)
    
    s = s.translate(E2W)
    s = re.sub(r"\s+", " ", s).strip()
    
    return s

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class PositionalEncoding(nn.Module):
    """Add positional information to embeddings"""
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Seq2SeqTransformer(nn.Module):
    """Transformer Encoder-Decoder for Urdu Chatbot"""
    def __init__(self, vocab, d_model=512, nhead=8, enc_layers=4, 
                 dec_layers=4, ff=2048, drop=0.1):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.pos_dec = PositionalEncoding(d_model)
        
        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ff,
            dropout=drop,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, vocab)
        
        self.config = {
            'vocab_size': vocab,
            'd_model': d_model,
            'nhead': nhead,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'ff': ff,
            'dropout': drop
        }

    def forward(self, src_ids, tgt_in_ids, src_kpm, tgt_kpm, tgt_causal):
        src = self.pos_enc(self.tok_emb(src_ids))
        tgt = self.pos_dec(self.tok_emb(tgt_in_ids))
        
        out = self.tf(
            src, tgt,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
            tgt_mask=tgt_causal
        )
        
        return self.fc_out(out)

# ============================================================
# DECODING FUNCTIONS
# ============================================================
@torch.no_grad()
def greedy_decode(model, sp, text, max_len=64, device='cpu', 
                  PAD=0, BOS=1, EOS=2):
    """Greedy decoding for fast inference"""
    model.eval()
    
    # Encode input
    src_tokens = sp.encode(normalize_urdu(text), out_type=int)
    if len(src_tokens) > 94:
        src_tokens = src_tokens[:94]
    
    src_ids = torch.tensor([[BOS] + src_tokens + [EOS]], 
                          dtype=torch.long, device=device)
    src_kpm = (src_ids == PAD)
    
    # Start with BOS token
    ys = torch.tensor([[BOS]], dtype=torch.long, device=device)
    
    # Generate tokens
    for _ in range(max_len):
        T = ys.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, 
                                      device=device), diagonal=1)
        logits = model(src_ids, ys, src_kpm, (ys == PAD), causal)
        next_id = logits[:, -1, :].argmax(-1)
        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
        
        if next_id.item() == EOS:
            break
    
    # Decode output
    out = ys[0, 1:]
    if (out == EOS).any():
        eos_idx = (out == EOS).nonzero(as_tuple=True)[0][0]
        out = out[:eos_idx]
    
    return sp.decode(out.tolist())


@torch.no_grad()
def beam_search_decode(model, sp, text, beam_width=4, max_len=64, 
                       device='cpu', PAD=0, BOS=1, EOS=2):
    """Beam search decoding for better quality"""
    model.eval()
    
    # Encode input
    src_tokens = sp.encode(normalize_urdu(text), out_type=int)
    if len(src_tokens) > 94:
        src_tokens = src_tokens[:94]
    
    src_ids = torch.tensor([[BOS] + src_tokens + [EOS]], 
                          dtype=torch.long, device=device)
    src_kpm = (src_ids == PAD)
    
    # Initialize beams
    beams = [(0.0, torch.tensor([[BOS]], device=device))]
    completed_beams = []
    
    for step in range(max_len):
        candidates = []
        
        for score, seq in beams:
            if seq[0, -1].item() == EOS:
                completed_beams.append((score / len(seq[0]), seq))
                continue
            
            T = seq.size(1)
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, 
                                          device=device), diagonal=1)
            logits = model(src_ids, seq, src_kpm, (seq == PAD), causal)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            
            topk_probs, topk_ids = torch.topk(log_probs, beam_width)
            
            for prob, token_id in zip(topk_probs[0], topk_ids[0]):
                new_seq = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], 
                                   dim=1)
                new_score = score + prob.item()
                candidates.append((new_score, new_seq))
        
        beams = sorted(candidates, key=lambda x: x[0], 
                      reverse=True)[:beam_width]
        
        if len(beams) == 0:
            break
    
    # Add remaining beams
    for score, seq in beams:
        completed_beams.append((score / len(seq[0]), seq))
    
    if not completed_beams:
        return ""
    
    # Get best sequence
    best_seq = max(completed_beams, key=lambda x: x[0])[1][0, 1:]
    
    if (best_seq == EOS).any():
        eos_idx = (best_seq == EOS).nonzero(as_tuple=True)[0][0]
        best_seq = best_seq[:eos_idx]
    
    return sp.decode(best_seq.tolist())

# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================
@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path, device):
    """Load model and tokenizer (cached)"""
    try:
        # Load tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        
        vocab_size = checkpoint.get('vocab_size', sp.get_piece_size())
        
        model = Seq2SeqTransformer(
            vocab=vocab_size,
            d_model=512,
            nhead=8,
            enc_layers=4,
            dec_layers=4,
            ff=2048,
            drop=0.1
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, sp, None
    except Exception as e:
        return None, None, str(e)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'message_count' not in st.session_state:
    st.session_state.message_count = 0

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ ترتیبات | Settings")
    
    # Model files
    st.markdown("#### 📁 Model Files")
    model_path = st.text_input(
        "Model Path (.pt)",
        value="best_bleu_urdu_chatbot.pt",
        help="Path to your trained model checkpoint"
    )
    
    tokenizer_path = st.text_input(
        "Tokenizer Path (.model)",
        value="spm/urdu.model",
        help="Path to your SentencePiece tokenizer"
    )
    
    # Device selection
    device = st.selectbox(
        "Device",
        ["cuda" if torch.cuda.is_available() else "cpu", "cpu"],
        help="Select computation device"
    )
    
    st.markdown("---")
    
    # Decoding settings
    st.markdown("#### 🎯 Decoding Strategy")
    decode_strategy = st.radio(
        "Strategy",
        ["Greedy", "Beam Search"],
        help="Greedy is faster, Beam Search gives better quality"
    )
    
    if decode_strategy == "Beam Search":
        beam_width = st.slider(
            "Beam Width",
            min_value=2,
            max_value=10,
            value=4,
            help="Higher values = better quality but slower"
        )
    else:
        beam_width = 1
    
    max_length = st.slider(
        "Max Length",
        min_value=32,
        max_value=128,
        value=64,
        step=8,
        help="Maximum tokens to generate"
    )
    
    st.markdown("---")
    
    # Actions
    st.markdown("#### 🔧 Actions")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.message_count = 0
        st.rerun()
    
    if st.button("💾 Export Chat", use_container_width=True):
        if st.session_state.chat_history:
            chat_json = json.dumps(
                st.session_state.chat_history,
                ensure_ascii=False,
                indent=2
            )
            st.download_button(
                "Download JSON",
                data=chat_json,
                file_name="urdu_chat_history.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("No chat history to export!")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("#### 📊 Statistics")
    st.metric("Total Messages", st.session_state.message_count)
    st.metric("Conversations", len(st.session_state.chat_history))

# ============================================================
# MAIN INTERFACE
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">💬 اردو چیٹ بوٹ</h1>
    <p style="margin:5px 0 0 0; font-size:14px;">
        Transformer-based Urdu Conversational AI
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("🔄 Loading model and tokenizer..."):
    model, sp, error = load_model_and_tokenizer(
        model_path, 
        tokenizer_path, 
        device
    )

if error:
    st.error(f"❌ Error loading model: {error}")
    st.info("""
    **Instructions:**
    1. Make sure your model file (.pt) is in the correct path
    2. Verify tokenizer file (.model) exists
    3. Check file paths in the sidebar
    """)
    st.stop()

st.success(f"✅ Model loaded successfully on {device}")

# Chat container
st.markdown("### 💭 گفتگو | Conversation")

chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.info("👋 سلام! آپ کی مدد کے لیے حاضر ہوں۔ کچھ پوچھیں!")
    else:
        for user_msg, bot_msg in st.session_state.chat_history:
            # User message
            st.markdown(f"""
            <div class="user-message rtl">
                <div class="message-label">آپ:</div>
                <div class="message-text">{user_msg}</div>
            </div>
            <div style="clear:both;"></div>
            """, unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f"""
            <div class="bot-message rtl">
                <div class="message-label">بوٹ:</div>
                <div class="message-text">{bot_msg}</div>
            </div>
            <div style="clear:both;"></div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input area
st.markdown("### ✍️ پیغام لکھیں | Write Message")

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_area(
        "Your message (اپنا پیغام)",
        height=100,
        placeholder="یہاں اردو میں ٹائپ کریں...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    send_button = st.button("📤 Send", use_container_width=True, type="primary")

# Process input
if send_button and user_input.strip():
    with st.spinner("🤔 جواب تیار ہو رہا ہے..."):
        try:
            # Generate response
            if decode_strategy == "Greedy":
                response = greedy_decode(
                    model, sp, user_input,
                    max_len=max_length,
                    device=device
                )
            else:
                response = beam_search_decode(
                    model, sp, user_input,
                    beam_width=beam_width,
                    max_len=max_length,
                    device=device
                )
            
            # Add to history
            st.session_state.chat_history.append((user_input, response))
            st.session_state.message_count += 2
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>🚀 Transformer Encoder-Decoder Architecture | Built with PyTorch & Streamlit</p>
    <p>📚 Trained on Urdu conversational data with SentencePiece tokenization</p>
</div>
""", unsafe_allow_html=True)
