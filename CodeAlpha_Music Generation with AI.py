
import os
import random
import collections

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from music21 import stream, note, chord, instrument, tempo, meter, midi




CLASSICAL_SEED = [
    # Simplified Bach-style motif
    ("C4",0.5),("D4",0.5),("E4",0.5),("F4",0.5),
    ("G4",1.0),("E4",0.5),("C4",0.5),
    ("D4",0.5),("E4",0.5),("F4",0.5),("G4",0.5),
    ("A4",1.0),("G4",0.5),("F4",0.5),
    ("E4",0.5),("F4",0.5),("G4",0.5),("A4",0.5),
    ("B4",1.0),("A4",0.5),("G4",0.5),
    ("G4",0.5),("A4",0.5),("B4",0.5),("C5",0.5),
    ("C5",2.0),
    # Second phrase
    ("E4",0.5),("D4",0.5),("C4",0.5),("B3",0.5),
    ("A3",1.0),("C4",0.5),("E4",0.5),
    ("F4",0.5),("E4",0.5),("D4",0.5),("C4",0.5),
    ("G4",2.0),
    ("C4",0.5),("E4",0.5),("G4",0.5),("E4",0.5),
    ("C4",2.0),
]

JAZZ_SEED = [
    # Blues-flavored ii–V–I in C
    ("D4",0.5),("F4",0.5),("A4",0.5),("C5",0.5),
    ("E4",0.5),("G4",0.5),("B4",0.5),("D5",0.5),
    ("G3",0.5),("B3",0.5),("D4",0.5),("F4",0.5),
    ("C4",1.0),("E4",0.5),("G4",0.5),
    # walking feel
    ("A3",0.5),("Bb3",0.5),("B3",0.5),("C4",0.5),
    ("D4",0.5),("Eb4",0.5),("E4",0.5),("F4",0.5),
    ("G4",1.0),("F4",0.5),("Eb4",0.5),
    ("D4",0.5),("C4",0.5),("Bb3",0.5),("A3",0.5),
    ("G3",2.0),
    # turnaround
    ("E4",0.5),("Eb4",0.5),("D4",0.5),("Db4",0.5),
    ("C4",2.0),
]



def seed_to_tokens(seed_data):
    """Convert (pitch, dur) pairs → list of string tokens like 'C4_0.5'."""
    return [f"{p}_{d}" for p, d in seed_data]


def build_vocab(token_lists):
    all_tokens = [t for lst in token_lists for t in lst]
    counts = collections.Counter(all_tokens)
    vocab = sorted(counts.keys())          # deterministic order
    tok2idx = {t: i for i, t in enumerate(vocab)}
    idx2tok = {i: t for t, i in tok2idx.items()}
    return tok2idx, idx2tok


def make_sequences(tokens, tok2idx, seq_len=16):
    """Sliding-window sequences for next-token prediction."""
    idxs = [tok2idx[t] for t in tokens]
    X, y = [], []
    for i in range(len(idxs) - seq_len):
        X.append(idxs[i : i + seq_len])
        y.append(idxs[i + seq_len])
    return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed    = nn.Embedding(vocab_size, embed_dim)
        self.lstm     = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout)
        self.fc       = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)                   # (B, T, E)
        out, hidden = self.lstm(x, hidden)  # (B, T, H)
        logits = self.fc(out[:, -1, :])     # last step → (B, V)
        return logits, hidden


def train(model, X, y, epochs=60, batch_size=32, lr=0.003):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"\nTraining on {device}  |  {len(X)} samples  |  vocab={model.fc.out_features}\n")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()
        if epoch % 10 == 0:
            avg = total_loss / len(dataset)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    model.eval()
    return model, device



def generate_tokens(model, device, seed_idxs, idx2tok, n_tokens=64,
                    temperature=0.85, seq_len=16):
    """Auto-regressively sample new tokens."""
    model.eval()
    generated = list(seed_idxs[-seq_len:])

    with torch.no_grad():
        for _ in range(n_tokens):
            inp = torch.tensor([generated[-seq_len:]], dtype=torch.long).to(device)
            logits, _ = model(inp)                   # (1, V)
            logits = logits / temperature
            probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            # nucleus (top-p) sampling for musicality
            sorted_idx = np.argsort(probs)[::-1]
            cumulative = 0.0
            top_idxs   = []
            for idx in sorted_idx:
                cumulative += probs[idx]
                top_idxs.append(idx)
                if cumulative >= 0.9:
                    break
            top_probs = probs[top_idxs]
            top_probs = top_probs / top_probs.sum()
            chosen = np.random.choice(top_idxs, p=top_probs)
            generated.append(int(chosen))

    return [idx2tok[i] for i in generated]



def tokens_to_midi(tokens, output_path="generated_music.mid",
                   bpm=120, style="classical"):
    """Convert token strings back to a music21 stream and save as MIDI."""
    s   = stream.Score()
    p   = stream.Part()
    p.append(meter.TimeSignature("4/4"))
    p.append(tempo.MetronomeMark(number=bpm))

    if style == "jazz":
        p.append(instrument.Piano())
    else:
        p.append(instrument.Piano())

    for token in tokens:
        try:
            pitch_str, dur_str = token.rsplit("_", 1)
            dur = float(dur_str)
            if "." in pitch_str and pitch_str[0].isalpha():
                # chord token like "C4.E4.G4"
                pitches = pitch_str.split(".")
                c = chord.Chord(pitches)
                c.quarterLength = dur
                p.append(c)
            else:
                n = note.Note(pitch_str)
                n.quarterLength = dur
                p.append(n)
        except Exception:
            # unknown token → quarter rest
            r = note.Rest()
            r.quarterLength = 1.0
            p.append(r)

    s.append(p)
    mf = midi.translate.music21ObjectToMidiFile(s)
    mf.open(output_path, "wb")
    mf.write()
    mf.close()
    print(f"\n✅  MIDI saved → {output_path}")
    return output_path



def midi_to_wav(midi_path):
    try:
        from midi2audio import FluidSynth
        wav_path = midi_path.replace(".mid", ".wav")
        fs = FluidSynth()
        fs.midi_to_audio(midi_path, wav_path)
        print(f"🎵  WAV saved  → {wav_path}")
        return wav_path
    except ImportError:
        print("ℹ️   midi2audio not installed – skipping WAV conversion.")
        print("    Install with:  pip install midi2audio  (also needs FluidSynth)")
    except Exception as e:
        print(f"⚠️   WAV conversion failed: {e}")
    return None



def main():
    SEQ_LEN    = 16
    EPOCHS     = 80
    GEN_TOKENS = 80
    TEMPERATURE= 0.85
    BPM        = 120
    STYLE      = "classical"   # "classical" or "jazz"
    OUTPUT     = "generated_music.mid"

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


    print("=" * 60)
    print("  AI Music Generator – LSTM on MIDI note sequences")
    print("=" * 60)

    seed_data = CLASSICAL_SEED + JAZZ_SEED          # combine both styles
    token_list = seed_to_tokens(seed_data)

    # Augment by repeating / transposing slightly (simple data augmentation)
    augmented = []
    for _ in range(8):                              # repeat corpus 8x
        augmented.extend(token_list)
    token_list = augmented

    tok2idx, idx2tok = build_vocab([token_list])
    vocab_size = len(tok2idx)
    print(f"\n📖  Vocabulary size : {vocab_size} unique tokens")
    print(f"📝  Total tokens    : {len(token_list)}")


    X, y = make_sequences(token_list, tok2idx, seq_len=SEQ_LEN)
    print(f"🔢  Training pairs  : {len(X)}")

    model = MusicLSTM(vocab_size=vocab_size,
                      embed_dim=64, hidden_dim=256,
                      num_layers=2, dropout=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠  Model params    : {total_params:,}")


    model, device = train(model, X, y, epochs=EPOCHS)

    print("\n🎼  Generating music …")
    # pick a seed window from classical or jazz
    seed_tokens = seed_to_tokens(CLASSICAL_SEED if STYLE == "classical" else JAZZ_SEED)
    seed_idxs   = [tok2idx[t] for t in seed_tokens if t in tok2idx]

    gen_tokens = generate_tokens(model, device, seed_idxs, idx2tok,
                                 n_tokens=GEN_TOKENS,
                                 temperature=TEMPERATURE,
                                 seq_len=SEQ_LEN)
    print(f"   Generated {len(gen_tokens)} tokens")
    print(f"   Sample: {gen_tokens[:8]} …")


    midi_path = tokens_to_midi(gen_tokens, output_path=OUTPUT,
                               bpm=BPM, style=STYLE)

    midi_to_wav(midi_path)

    print("\n🎉  Done!  Open the .mid file in any DAW, MuseScore, or media player.")


if __name__ == "__main__":
    main()