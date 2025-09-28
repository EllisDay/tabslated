# app.py — minimal, from-scratch, chord detection via Sonic Annotator + Chordino
import os, io, subprocess, tempfile, pathlib
import numpy as np
#import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse


# ====== Robust chord post-processing & pattern smoothing ======
import re, numpy as np
from collections import Counter, defaultdict

# ====== Whisper (self-hosted, free) ======
from faster_whisper import WhisperModel

# Choose model via env var WHISPER_MODEL, fallback to something that runs on CPU.
# Options in increasing accuracy/size: tiny, base, small, medium, large-v3
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
# int8 is fast on CPU; use "float16"/"float32" if you have a GPU and want max accuracy.
WHISPER = None
def get_whisper():
    global WHISPER
    if WHISPER is None:
        WHISPER = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    return WHISPER


def _lrc_ts(t: float | None) -> str:
    if t is None: t = 0.0
    m = int(t // 60); s = t - 60*m
    return f"[{m:02d}:{s:05.2f}]"

def whisper_to_lrc(audio_path: str, per_word=False) -> str:
    """
    Transcribe 'audio_path' with faster-whisper and return LRC text.
    - per_word=False: one LRC line per ASR segment (good default)
    - per_word=True : one LRC line per word (karaoke-style; noisier on singing)
    """
    wm = get_whisper()
    segments, info = wm.transcribe(
        audio_path,
        beam_size = 5,
        vad_filter = True,
        word_timestamps = True
    )
    if per_word:
        lines = []
        for seg in segments:
            for w in (seg.words or []):
                txt = (w.word or "").strip()
                if txt:
                    lines.append(f"{_lrc_ts(w.start)} {txt}")
        return "\n".join(lines)
    else:
        lines = []
        for seg in segments:
            txt = (seg.text or "").strip()
            if txt:
                lines.append(f"{_lrc_ts(seg.start)} {txt}")
        return "\n".join(lines)


# ---------- Key detection + diatonic correction ----------
import re
try:
    import librosa
except Exception:
    librosa = None

PC_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
NOTE_TO_PC = {
    "C":0,"B#":0,
    "C#":1,"Db":1,
    "D":2,
    "D#":3,"Eb":3,
    "E":4,"Fb":4,
    "F":5,"E#":5,
    "F#":6,"Gb":6,
    "G":7,
    "G#":8,"Ab":8,
    "A":9,
    "A#":10,"Bb":10,
    "B":11,"Cb":11,
}

# Krumhansl key profiles (normalized)
_KR_MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
_KR_MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)
_KR_MAJ /= np.linalg.norm(_KR_MAJ); _KR_MIN /= np.linalg.norm(_KR_MIN)

def estimate_key_krumhansl(wav_path: str, sr=22050):
    """
    Estimate global key by correlating mean chroma with Krumhansl profiles.
    Returns (tonic_pc[0..11], mode_str 'maj'|'min')
    """
    if librosa is None:
        return None, None
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)   # (12, T)
    prof = chroma.mean(axis=1) + 1e-9
    prof /= np.linalg.norm(prof)
    best = (-1e9, 0, 'maj')
    for tonic in range(12):
        rot = np.roll(prof, -tonic)
        cmaj = float(np.dot(rot, _KR_MAJ))
        cmin = float(np.dot(rot, _KR_MIN))
        if cmaj > best[0]: best = (cmaj, tonic, 'maj')
        if cmin > best[0]: best = (cmin, tonic, 'min')
    return best[1], best[2]

def diatonic_quality_for(root_pc: int, tonic_pc: int, mode: str):
    """Return 'maj'|'min'|'dim' or None if root not diatonic in that mode."""
    deg = (root_pc - tonic_pc) % 12
    if mode == 'maj':
        mapping = {0:'maj', 2:'min', 4:'min', 5:'maj', 7:'maj', 9:'min', 11:'dim'}
        return mapping.get(deg)
    else:  # natural minor; close enough for quality correction
        mapping = {0:'min', 2:'dim', 3:'maj', 5:'min', 7:'min', 8:'maj', 10:'maj'}
        return mapping.get(deg)

_CHORD_RE = re.compile(r'^([A-G][b#]?)(m|dim|aug)?$')

def key_aware_correct(segments, tonic_pc, mode,
                      drop_out_of_scale_if_short=True,
                      short_thresh=1.2):
    """
    Snap each chord's quality to the diatonic quality for the detected key.
    Optionally drop out-of-scale short anomalies.
    """
    if tonic_pc is None or mode not in ('maj','min'):
        return segments
    out = []
    for s in segments:
        name = s["chord"]
        if name == "N.C.":
            out.append(s); continue
        m = _CHORD_RE.match(name)
        if not m:
            out.append(s); continue
        root, qual = m.group(1), (m.group(2) or '')
        root_pc = NOTE_TO_PC.get(root)
        if root_pc is None:
            out.append(s); continue
        target = diatonic_quality_for(root_pc, tonic_pc, mode)
        if target:
            # force diatonic quality
            new = root + ('m' if target=='min' else ('dim' if target=='dim' else ''))
            out.append({**s, "chord": new})
        else:
            # root not in scale
            if drop_out_of_scale_if_short and (s["end"]-s["start"]) < short_thresh:
                out.append({**s, "chord": "N.C."})
            else:
                out.append(s)
    return _merge_adjacent(out)


# ---- Simple chord fallback using librosa chroma (no Sonic needed) ----
def fallback_chords_librosa(wav_path: str,
                            sr=22050,
                            hop_s=0.50,
                            energy_thresh_db=-60.0):
    """
    Very simple chord estimator:
      - chroma_CENS
      - correlate vs 12 major / 12 minor triad templates
      - choose best each hop_s seconds
      - suppress very low-energy frames to N.C.
      - return merged segments [{start,end,chord}]
    """
    if librosa is None:
        return []

    # Load mono
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    hop_length = max(1, int(sr * hop_s))

    # Chroma (12, T)
    C = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    T = C.shape[1]
    if T == 0:
        return []

    # Frame times
    times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)

    # Rough loudness per frame (to mask silence)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True).flatten()
    rms_db = librosa.amplitude_to_db(np.maximum(1e-10, rms), ref=1.0)

    # Build templates (12 majors, 12 minors)
    baseM = np.zeros(12); baseM[0]=baseM[4]=baseM[7]=1.0   # C major: 0,4,7
    basem = np.zeros(12); basem[0]=basem[3]=basem[7]=1.0   # C minor: 0,3,7
    maj_templates = np.stack([np.roll(baseM, k) for k in range(12)])  # (12,12)
    min_templates = np.stack([np.roll(basem, k) for k in range(12)])  # (12,12)

    # Normalize chroma columns
    Cn = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)

    # Scores
    maj_scores = maj_templates @ Cn  # (12,T)
    min_scores = min_templates @ Cn  # (12,T)

    labels = []
    for t in range(T):
        if rms_db[t] < energy_thresh_db:
            labels.append("N.C.")
            continue
        jm = int(np.argmax(maj_scores[:, t]))
        jn = int(np.argmax(min_scores[:, t]))
        if maj_scores[jm, t] >= min_scores[jn, t]:
            labels.append(PC_NAMES[jm])
        else:
            labels.append(PC_NAMES[jn] + "m")

    # Framewise → segments
    segs = []
    cur = labels[0]; start = float(times[0]) if T else 0.0
    for i in range(1, T):
        if labels[i] != cur:
            end = float(times[i])
            if cur != "N.C.":
                segs.append({"start": round(start,3), "end": round(end,3), "chord": cur})
            cur = labels[i]; start = float(times[i])
    tail = float(times[-1] + hop_s if T else start + hop_s)
    if cur != "N.C.":
        segs.append({"start": round(start,3), "end": round(tail,3), "chord": cur})

    # Reuse your cleanup
    return _prune_blips(_merge_adjacent(segs))



# ---- TUNABLES ----
HOP              = 0.50     # seconds per token for rasterization
MODE_WIN         = 5        # odd number; token mode filter window
MIN_CHORD_DUR    = 1.20     # < this = treat as blip (unless slot expects it)
MIN_NC_DUR       = 0.60     # tiny N.C. gaps are dropped/bridged
MIN_DIM_DUR      = 2.00     # short dim/aug hits are likely noise
K_CANDIDATES     = [3,4,5,6,7,8]  # pattern lengths to try
MIN_OCC          = 2        # pattern must repeat at least this many times
MIN_SEP_TOK      = 8        # min token separation between two occurrences
MAX_NC_RATIO     = 0.50     # skip windows that are >50% N.C.
ALT_THRESH       = 0.35     # allow alternates at slot if they appear ≥35% there
STRICT_ENFORCE   = True     # when True, coerce anything inside a repeated window to consensus
KEEP_OUTSIDE     = True     # keep (smoothed) chords outside repeated windows

# ---- Label simplification ----
# Normalize labels like "Ebmaj7", "Bb6", "Gdim", "Cm7", "A:7" → triads: "Eb", "Bb", "Gdim", "Cm", "A"
_TRIAD_RE = re.compile(r'^([A-G][b#]?)(?::([A-Za-z0-9+\-]+))?$')

def simplify_label(raw: str) -> str:
    """
    Map a Chordino label to a stable triad-ish symbol.
    - N -> N.C.
    - min/min7/min9 -> m
    - maj/maj7/6/9/add/sus/7 -> major triad (drop color)
    - dim/°/o/aug/+ stay as 'dim'/'aug' (we'll suppress if too short)
    """
    lab = raw.strip().strip('"').strip()
    if lab == 'N':
        return 'N.C.'
    m = _TRIAD_RE.match(lab)
    if not m:
        # bare root like "Eb" or "Cm7" without colon might still be fine:
        # try manual parse
        if lab.endswith('m'):
            return lab  # "Cm"
        return lab  # "Eb", "Bb6" etc; handled below
    root = m.group(1)
    qual = (m.group(2) or 'maj').lower()

    # minor family
    if qual.startswith('min') or qual.startswith('m'):
        return root + 'm'

    # diminished / augmented treated explicitly
    if 'dim' in qual or '°' in qual or 'o' in qual:
        return root + 'dim'
    if 'aug' in qual or '+' in qual:
        return root + 'aug'

    # dominant 7 / maj7 / 6 / 9 / add / sus etc → collapse to major triad
    return root

def collapse_colors(name: str) -> str:
    """
    Further collapse text labels like 'Bb6','Ebmaj7','G7','Fsus4' → 'Bb','Eb','G','F'
    Keep 'Xm' minor, and 'Xdim','Xaug' as-is.
    """
    if name == 'N.C.': return name
    if name.endswith('m'): return name
    if name.endswith('dim') or name.endswith('aug'): return name
    # drop trailing color strings
    # common endings: maj7, maj9, 6, 7, 9, 11, 13, sus2, sus4, add9, etc.
    base = re.match(r'^([A-G][b#]?)(?:.*)?$', name)
    return base.group(1) if base else name

# ---- Parse LAB with 2-col *or* 3-col support, then simplify ----
_LAB3 = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+(.+?)\s*$')
_LAB2 = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?)\s+(.+?)\s*$')

def parse_lab(lab_text: str):
    # Try 3-col first (start end label)
    segs = []
    for line in (lab_text or "").splitlines():
        m = _LAB3.match(line)
        if m:
            s = float(m.group(1)); e = float(m.group(2)); raw = m.group(3).split()[0]
            simp = collapse_colors(simplify_label(raw))
            segs.append({"start": round(s,3), "end": round(e,3), "chord": simp})
    if segs:
        return segs

    # Fallback to 2-col events (time label) -> stitch into segments
    events = []
    for line in (lab_text or "").splitlines():
        m = _LAB2.match(line)
        if not m:
            continue
        t = float(m.group(1)); raw = m.group(2).split()[0]
        simp = collapse_colors(simplify_label(raw))
        events.append((t, simp))
    if len(events) < 2:
        return []
    out = []
    for i in range(len(events)-1):
        t0, lab0 = events[i]; t1, _ = events[i+1]
        if lab0 == 'N.C.':
            continue
        out.append({"start": round(t0,3), "end": round(t1,3), "chord": lab0})
    return out

_TRIAD_RE = re.compile(r'^([A-G][b#]?)(?::([A-Za-z0-9+\-]+))?$')
def _simplify(raw: str) -> str:
    lab = raw.strip().strip('"').strip()
    if lab == 'N': return 'N.C.'
    m = _TRIAD_RE.match(lab)
    if m:
        root = m.group(1)
        qual = (m.group(2) or 'maj').lower()
        if qual.startswith('min') or qual.startswith('m'): return root + 'm'
        if 'dim' in qual or '°' in qual or 'o' in qual:   return root + 'dim'
        if 'aug' in qual or '+' in qual:                   return root + 'aug'
        return root
    # fallback (e.g., "Bb6", "Cm7")
    if lab.endswith('m'): return lab
    m = re.match(r'^([A-G][b#]?)', lab)
    return m.group(1) if m else lab

def _merge_adjacent(segs):
    if not segs: return []
    out = [segs[0].copy()]
    for s in segs[1:]:
        if s["chord"] == out[-1]["chord"] and abs(s["start"] - out[-1]["end"]) < 1e-3:
            out[-1]["end"] = s["end"]
        else:
            out.append(s.copy())
    return out

def _simplify_segments(segs):
    out = []
    for s in segs:
        out.append({"start": s["start"], "end": s["end"], "chord": _simplify(s["chord"])})
    return _merge_adjacent(out)


def _prune_blips(segs):
    if not segs: return []
    S = _merge_adjacent(segs)

    # bridge tiny N.C. between identical neighbors
    i = 1
    while i < len(S)-1:
        a, b, c = S[i-1], S[i], S[i+1]
        if b["chord"] == "N.C." and (b["end"]-b["start"]) < MIN_NC_DUR and a["chord"] == c["chord"]:
            a["end"] = c["start"]
            S.pop(i); S = _merge_adjacent(S); continue
        i += 1

    # short dim/aug → N.C.
    for s in S:
        if (s["chord"].endswith("dim") or s["chord"].endswith("aug")) and (s["end"]-s["start"]) < MIN_DIM_DUR:
            s["chord"] = "N.C."
    S = _merge_adjacent(S)

    # short non-N.C. → absorb into longer neighbor
    out = []
    for idx, s in enumerate(S):
        d = s["end"] - s["start"]
        if s["chord"] != "N.C." and d < MIN_CHORD_DUR:
            prev = out[-1] if out else None
            nxt  = S[idx+1] if idx+1 < len(S) else None
            if prev and nxt and prev["chord"] == nxt["chord"]:
                s["chord"] = prev["chord"]
            elif prev and (not nxt or (prev["end"]-prev["start"]) >= (nxt["end"]-nxt["start"])):
                s["chord"] = prev["chord"]
            elif nxt:
                s["chord"] = nxt["chord"]
            else:
                s["chord"] = "N.C."
        out.append(s)
    return _merge_adjacent(out)

# ----- tokenization -----
def _rasterize(segs, hop=HOP):
    T = segs[-1]["end"] if segs else 0.0
    if T <= 0: return [], np.array([])
    times = np.arange(0.0, T, hop)
    toks = []
    j = 0
    for t in times:
        while j < len(segs) and segs[j]["end"] <= t: j += 1
        if j < len(segs) and segs[j]["start"] <= t < segs[j]["end"]:
            toks.append(segs[j]["chord"])
        else:
            toks.append("N.C.")
    return toks, times

def _mode_filter(tokens, win=MODE_WIN):
    if win < 3 or win % 2 == 0: return tokens[:]
    out = tokens[:]; half = win//2
    for i in range(len(tokens)):
        a = max(0, i-half); b = min(len(tokens), i+half+1)
        sub = tokens[a:b]
        counts = Counter(sub)
        # prefer non-N.C. in ties
        best = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]!="N.C."), reverse=True)[0][0]
        out[i] = best
    return out

def _derasterize(tokens, times):
    if not tokens: return []
    segs = []
    cur = tokens[0]; start = float(times[0])
    for i in range(1, len(tokens)):
        if tokens[i] != cur:
            segs.append({"start": round(start,3), "end": round(float(times[i]),3), "chord": cur})
            cur = tokens[i]; start = float(times[i])
    tail = float(times[-1] + (times[1]-times[0]) if len(times)>1 else start+HOP)
    segs.append({"start": round(start,3), "end": round(tail,3), "chord": cur})
    # drop tiny N.C.
    segs = [s for s in segs if s["chord"]!="N.C." or (s["end"]-s["start"]) >= MIN_NC_DUR]
    return _merge_adjacent(segs)


# ---- Repeated-section detection ----
# ----- find best repeating window (variable k) -----
def _nonoverlapping_starts(all_starts, min_sep):
    sel = []; last = -10**9
    for s in sorted(all_starts):
        if s - last >= min_sep:
            sel.append(s); last = s
    return sel

def _score_window(window, starts):
    # coverage * repetition, penalize N.C.-heavy windows
    nc_ratio = window.count("N.C.")/len(window)
    unique = len({x for x in window if x!="N.C."})
    return (len(starts) * len(window) * max(0.0, 1.0 - nc_ratio)) + unique

def find_best_pattern(tokens, k_list=K_CANDIDATES, min_occ=MIN_OCC, min_sep=MIN_SEP_TOK):
    """Return (pattern_tokens, starts) or (None, [])."""
    best = (None, [])
    best_score = -1
    for k in k_list:
        if len(tokens) < k: continue
        table = defaultdict(list)
        for i in range(len(tokens)-k+1):
            win = tuple(tokens[i:i+k])
            if win.count("N.C.")/k > MAX_NC_RATIO:  # too many no-chords → skip
                continue
            if len({x for x in win if x!="N.C."}) < 2:  # need at least 2 distinct chords
                continue
            table[win].append(i)
        for win, idxs in table.items():
            starts = _nonoverlapping_starts(idxs, min_sep)
            if len(starts) < min_occ:
                continue
            sc = _score_window(list(win), starts)
            if sc > best_score:
                best_score = sc
                best = (list(win), starts)
    return best

def consensus_for_pattern(tokens, starts, k):
    """Per-slot consensus & alternates across all occurrences."""
    cons = []; alts = []
    for off in range(k):
        votes = [tokens[s+off] for s in starts if s+off < len(tokens)]
        votes = [v for v in votes if v!="N.C."]
        if votes:
            c = Counter(votes)
            top = c.most_common(1)[0][0]
            cons.append(top)
            # alternates above threshold
            allowed = {lab for lab, n in c.items() if n/sum(c.values()) >= ALT_THRESH}
            alts.append(allowed)
        else:
            cons.append("N.C."); alts.append(set())
    return cons, alts  # lists of length k

def enforce_pattern_variable(tokens, times, pattern, starts, strict=STRICT_ENFORCE):
    """Coerce tokens inside the repeated windows to the consensus pattern (with allowed alternates)."""
    k = len(pattern)
    cons, alts = consensus_for_pattern(tokens, starts, k)
    toks = tokens[:]
    covered = np.zeros(len(tokens), dtype=bool)

    for s in starts:
        for off in range(k):
            i = s+off
            if i >= len(toks): break
            covered[i] = True
            want = cons[off]
            allowed = alts[off] | {want}
            if toks[i] in allowed:
                continue
            if strict or toks[i]=="N.C.":
                toks[i] = want

    # light smoothing inside covered regions
    toks = _mode_filter(toks, win=MODE_WIN)

    # outside regions: keep (optionally mode-filter)
    if KEEP_OUTSIDE:
        return _derasterize(toks, times), covered
    else:
        # zero out outside (optional path)
        for i, c in enumerate(covered):
            if not c: toks[i] = "N.C."
        return _derasterize(toks, times), covered

def postprocess_variable_pattern(raw_segments):
    """
    Full pipeline for variable-length patterns.
    Returns final_segments, pattern_tokens, section_ranges
    """
    simp = _simplify_segments(raw_segments)
    base = _prune_blips(simp)
    tokens, times = _rasterize(base)
    if not tokens:
        return base, [], []

    tokens = _mode_filter(tokens, win=MODE_WIN)

    pattern, starts = find_best_pattern(tokens)
    if not pattern:
        # no solid repeats found → just derasterize the smoothed tokens
        rough = _derasterize(tokens, times)
        return _prune_blips(rough), [], []

    enforced, covered_mask = enforce_pattern_variable(tokens, times, pattern, starts, strict=STRICT_ENFORCE)
    final = _prune_blips(enforced)

    # section time ranges for UI (merge overlapping windows)
    secs = []
    for s in starts:
        t0 = float(times[s]); t1 = float(times[min(s+len(pattern), len(times)-1)])
        secs.append({"start": round(t0,3), "end": round(t1,3)})
    # merge overlapping/adjacent sections
    secs.sort(key=lambda x: x["start"])
    merged = []
    for sec in secs:
        if not merged or sec["start"] > merged[-1]["end"] + 0.01:
            merged.append(sec)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], sec["end"])

    return final, pattern, merged


# ===== Lyrics parsing & chord-over-lyrics alignment =====
import re

# [mm:ss.xx] or [m:ss.x] style tags
_LRC_TIME = re.compile(r"\[(\d{1,2}):(\d{2})(?:\.(\d{1,2}))?\]")

def parse_lrc(lrc_text: str):
    """
    Parse LRC (or plain text) into a list of lines:
      [{"start": float|None, "end": float|None, "text": str}, ...]
    - Accepts multiple timestamps on one line (uses the earliest as the start).
    - Leaves "end" = None; caller typically fills end from the next line or song end.
    - If a line has no timestamps, we keep start=None and end=None (handled later).
    """
    out = []
    for raw in (lrc_text or "").splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        times = list(_LRC_TIME.finditer(line))
        # remove all [..] tags from the visible text
        text = _LRC_TIME.sub("", line).strip()
        if not times:
            # plain line (no timing)
            if text:
                out.append({"start": None, "end": None, "text": text})
            continue

        # convert all tags on the line, keep earliest as start
        starts = []
        for m in times:
            mm = int(m.group(1))
            ss = int(m.group(2))
            frac = m.group(3)
            if frac is None:
                t = mm*60 + ss
            else:
                # .x or .xx
                scale = 10 if len(frac) == 1 else 100
                t = mm*60 + ss + int(frac)/scale
            starts.append(float(t))
        if text:
            out.append({"start": min(starts), "end": None, "text": text})

    # Fill end times from the next line's start when both are known
    for i in range(len(out)-1):
        if out[i]["start"] is not None and out[i+1]["start"] is not None:
            out[i]["end"] = out[i+1]["start"]
    return out

def interpolate_words(line):
    """
    Given a line dict {"start","end","text"}, return a list of (word, time_or_None).
    If start/end missing or invalid, we return (word, None).
    Otherwise, we spread words uniformly in [start, end).
    """
    words = line["text"].split()
    if not words:
        return []
    t0, t1 = line.get("start"), line.get("end")
    if t0 is None or t1 is None or t1 <= t0:
        return [(w, None) for w in words]
    if len(words) == 1:
        return [(words[0], float(t0))]
    span = float(t1 - t0)
    step = span / (len(words)-1)
    return [(w, float(t0) + i*step) for i, w in enumerate(words)]

def align_chords_to_words(segments, lrc_lines):
    """
    Place chord names above the first character of the word whose time falls inside that chord span.

    segments: [{start: float, end: float, chord: str}, ...]
    lrc_lines: result of parse_lrc()

    Returns a list of per-line dicts:
      {
        "lyrics": "<the line text>",
        "placed": [(column_index, "ChordName"), ...],
        "mono":   "<chord row>\\n<lyrics row>"  # for <pre> monospace rendering
      }
    """
    # Safety: sort segments
    segs = sorted(segments, key=lambda s: (s["start"], s["end"]))

    def chord_at(t):
        if t is None:
            return None
        # Linear scan is fine for small N; swap for bisect if needed
        for s in segs:
            if s["start"] <= t < s["end"]:
                return s["chord"]
        return None

    aligned = []
    for line in lrc_lines:
        words_with_t = interpolate_words(line)
        lyrics_text = " ".join(w for w, _ in words_with_t)
        if not lyrics_text:
            aligned.append({"lyrics": "", "placed": [], "mono": ""})
            continue

        placed = []  # (column_index, chord)
        col = 0
        last = None
        for idx, (w, t) in enumerate(words_with_t):
            ch = chord_at(t)
            # Only place when chord changes and is not N.C.
            if ch and ch != "N.C." and ch != last:
                placed.append((col, ch))
            last = ch
            # advance position: word + one space (except after last word)
            col += len(w) + (1 if idx < len(words_with_t)-1 else 0)

        # Build chord row in monospace to align above lyrics_text
        chord_row = [" "] * len(lyrics_text)
        for pos, ch in placed:
            for i, c in enumerate(ch):
                j = pos + i
                if 0 <= j < len(chord_row):
                    chord_row[j] = c

        aligned.append({
            "lyrics": lyrics_text,
            "placed": placed,
            "mono": "".join(chord_row) + "\n" + lyrics_text
        })

    return aligned


# === EDIT THESE PATHS ===
SONIC = os.environ.get(
    "SONIC",
    r"C:\Users\ellis_59031kw\sonic-annotator\sonic-annotator-win64\sonic-annotator.exe"
)
VAMP_PATH = os.environ.get(
    "VAMP_PATH",
    r"C:\Users\ellis_59031kw\OneDrive\Desktop\vamp-plugins\Vamp Plugins;C:\ Program Files\Vamp Plugins"
)
DISABLE_SONIC = os.environ.get("DISABLE_SONIC", "1") == "1"

# (You can also use: r"C:\Program Files\Vamp Plugins" if that’s where the DLL is)
# ========================

VERSION = "SquareOne v1"

app = FastAPI(title=f"Chord Detector ({VERSION})")

# add near your imports
from google.cloud import storage
from google.oauth2 import service_account
import datetime, json

GCS_BUCKET = os.environ.get("GCS_BUCKET")  # set in Cloud Run
SIGN_URL_EXPIRE_MIN = 15

def _storage_client():
    # uses default creds in Cloud Run
    return storage.Client()

@app.post("/upload-url")
def get_upload_url():
    """
    Returns a resumable signed URL for uploading a WAV directly to GCS.
    Client does a PUT to this URL (or uses x-goog-resumable).
    """
    fn = f"uploads/{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{os.urandom(4).hex()}.wav"
    bucket = _storage_client().bucket(GCS_BUCKET)
    blob = bucket.blob(fn)

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=SIGN_URL_EXPIRE_MIN),
        method="PUT",
        content_type="audio/wav",
        headers={"x-goog-resumable": "start"},
    )
    return {"ok": True, "bucket": GCS_BUCKET, "object": fn, "upload_url": url}


import tempfile

def _download_gcs_to_temp(bucket_name: str, object_name: str) -> str:
    client = _storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    blob.download_to_filename(path)
    return path

from fastapi import Body

@app.post("/analyze_gcs")
def analyze_gcs(
    ref: dict = Body(...),  # { "bucket": "...", "object": "...", "lyrics_lrc": "...", "lyrics_auto": "true/false" }
):
    wav = None
    try:
        bucket = ref.get("bucket") or GCS_BUCKET
        obj = ref["object"]
        lyrics_lrc = (ref.get("lyrics_lrc") or "").strip()
        lyrics_auto = str(ref.get("lyrics_auto") or "").lower() == "true"

        wav = _download_gcs_to_temp(bucket, obj)

        # ---- chord detection (do NOT fail if Sonic missing) ----
        segments_error = None
        final, pattern, sections = [], [], []
        try:
            lab = run_sonic_simplechord(wav)  # works if Sonic in Docker image
            raw_segments = parse_lab(lab)
            final, pattern, sections = postprocess_variable_pattern(raw_segments)
        except Exception as e:
            segments_error = f"Chord detection unavailable: {type(e).__name__}: {e}"

        # ---- key detection ----
        det_key = None
        if final:
            try:
                tonic_pc, mode = estimate_key_krumhansl(wav)
                if tonic_pc is not None:
                    det_key = f"{PC_NAMES[tonic_pc]} {'major' if mode == 'maj' else 'minor'}"
                    final = key_aware_correct(final, tonic_pc, mode,
                                              drop_out_of_scale_if_short=True, short_thresh=1.2)
            except Exception:
                pass

        # ---- lyrics (ASR only if requested and empty) ----
        lrc_text = lyrics_lrc
        asr_error = None
        if not lrc_text and lyrics_auto:
            try:
                lrc_text = whisper_to_lrc(wav, per_word=False)
            except Exception as e:
                asr_error = f"Auto-lyrics failed: {type(e).__name__}: {e}"
                lrc_text = ""

        aligned = []
        if lrc_text:
            lines = parse_lrc(lrc_text)
            if not any(ln.get("start") is not None for ln in lines):
                total = float(final[-1]["end"]) if final else 0.0
                if total <= 0 and librosa is not None:
                    try:
                        y, sr = librosa.load(wav, sr=None, mono=True)
                        total = len(y) / float(sr)
                    except Exception:
                        total = 0.0
                n = max(1, len(lines))
                step = (total or n) / n
                for i, ln in enumerate(lines):
                    ln["start"] = i*step
                    ln["end"] = (i+1)*step if i < n-1 else total

            song_end = float(final[-1]["end"]) if final else (lines[-1]["end"] if lines and lines[-1].get("end") else None)
            for i, ln in enumerate(lines):
                if ln.get("start") is not None and ln.get("end") is None:
                    ln["end"] = lines[i+1]["start"] if i+1 < len(lines) and lines[i+1].get("start") is not None else song_end

            aligned = align_chords_to_words(final, lines)

        return JSONResponse({
            "ok": True,
            "gcs_object": obj,
            "detected_key": det_key,
            "detected_pattern": pattern,
            "sections": sections,
            "segments": final,
            "segments_error": segments_error,
            "lyrics_lrc": lrc_text or None,
            "aligned": aligned,
            "asr_error": asr_error,
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    finally:
        try:
            if wav and os.path.exists(wav):
                os.remove(wav)
        except Exception:
            pass



from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tabslated.web.app",   # your Firebase Hosting site
        "https://api.tabslated.com",   # your new API domain
        "http://localhost:3000",       # optional local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Core helpers -----
import tempfile

# 1) Run SA -> Chordino, capture LAB on stdout (no -o)
def run_sonic_simplechord(wav_path: str) -> str:
    if DISABLE_SONIC:
        raise RuntimeError("Chord detection temporarily disabled (DISABLE_SONIC=1).")
    if not os.path.isfile(SONIC):
        raise RuntimeError(f"Sonic Annotator not found at: {SONIC}")
    env = os.environ.copy()
    env["VAMP_PATH"] = VAMP_PATH
    cmd = [
        SONIC,
        "-d", "vamp:nnls-chroma:chordino:simplechord",
        "-w", "lab", "--lab-stdout",
        wav_path.replace("\\", "/"),
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="ignore")
    if p.returncode != 0:
        raise RuntimeError("Sonic Annotator failed:\n" + (p.stderr or p.stdout or ""))
    return p.stdout


# 2) Robust LAB parser that ignores any non-LAB lines
import re

# 3-col: start end label
_LAB3 = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+(.+?)\s*$')
# 2-col: time label  (what your smoketest shows)
_LAB2 = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?)\s+(.+?)\s*$')

def _norm_label(raw: str) -> str:
    lab = raw.strip().strip('"').strip()
    if lab == 'N':
        return 'N.C.'
    # Examples: C:maj, A:min, G:7, F
    parts = lab.split(':', 1)
    root = parts[0]
    qual  = parts[1] if len(parts) > 1 else 'maj'
    return root + ('m' if qual.startswith('min') else '')

def parse_lab(lab_text: str):
    """
    Parse Sonic Annotator LAB for Chordino:
    - If lines are 'start end label' -> return segments directly.
    - If lines are 'time label'      -> convert consecutive events to segments.
    Returns list of dicts: [{start, end, chord}, ...]
    """
    lines = (lab_text or "").splitlines()

    # Try 3-column first
    segs_3 = []
    for line in lines:
        m = _LAB3.match(line)
        if m:
            s = float(m.group(1)); e = float(m.group(2)); raw = m.group(3).split()[0]
            segs_3.append({"start": round(s,3), "end": round(e,3), "chord": _norm_label(raw)})
    if segs_3:
        return segs_3

    # Fallback: 2-column events  -> stitch into segments by taking [t[i], t[i+1]) with label[i]
    events = []
    for line in lines:
        m = _LAB2.match(line)
        if not m:
            continue  # skip any log noise
        t = float(m.group(1)); raw = m.group(2).split()[0]
        events.append((t, _norm_label(raw)))

    if len(events) < 2:
        return []  # not enough info to form a segment

    segs = []
    for i in range(len(events)-1):
        t0, lab0 = events[i]
        t1, _    = events[i+1]
        # skip leading/trailing/adjacent N.C. boundaries
        if lab0 == 'N.C.':
            continue
        segs.append({"start": round(t0,3), "end": round(t1,3), "chord": lab0})
    return segs


def merge_segments(segs, min_dur=0.6):
    """Merge consecutive identical chords; turn very short non-N.C. blips into N.C."""
    if not segs:
        return segs
    merged = []
    cur = segs[0].copy()
    for s in segs[1:]:
        if s["chord"] == cur["chord"] and abs(s["start"] - cur["end"]) < 0.02:
            cur["end"] = s["end"]
        else:
            if (cur["end"] - cur["start"]) < min_dur and cur["chord"] != "N.C.":
                cur["chord"] = "N.C."
            merged.append(cur)
            cur = s.copy()
    if (cur["end"] - cur["start"]) < min_dur and cur["chord"] != "N.C.":
        cur["chord"] = "N.C."
    merged.append(cur)
    return merged

def is_wav_bytes(raw: bytes) -> bool:
    # RIFF....WAVE
    return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"

def write_wav_from_bytes(raw: bytes) -> str:
    """Save uploaded bytes as a .wav file. We require WAV to keep things simple & robust."""
    if not is_wav_bytes(raw):
        raise RuntimeError("Please upload a WAV file (PCM/float). You can convert in any editor (e.g., Audacity).")
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    with open(path, "wb") as f:
        f.write(raw)
    return path

# ----- Self-test: synthesize a C major chord -----
# remove: import soundfile as sf
from scipy.io.wavfile import write as wavwrite

def synth_c_major_wav(seconds=3.0, sr=44100) -> str:
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False, dtype=np.float32)
    freqs = [261.63, 329.63, 392.00]  # C4, E4, G4
    y = sum(0.33*np.sin(2*np.pi*f*t).astype(np.float32) for f in freqs)
    Nf = int(sr*0.02)
    fade = np.linspace(0, 1, Nf, dtype=np.float32)
    y[:Nf] *= fade; y[-Nf:] *= fade[::-1]
    y /= max(1e-9, np.abs(y).max())
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    wavwrite(path, sr, (y * 32767).astype(np.int16))  # 16-bit PCM
    return path

# ----- Routes -----
@app.get("/", response_class=HTMLResponse)
def home():
    html = """
<!doctype html>
<meta charset="utf-8"/>
<title>🎸 Tabslated - Guitar Tab Generator 1 🎸</title>
<body style="font-family:system-ui;background:#0b0f19;color:#e8eef8">
  <div style="max-width:900px;margin:2rem auto">
    <h1>🎸 Tabslated - Guitar Tab Generator 1 🎸</h1>
    <p style="opacity:.8">Upload a <strong>WAV</strong>. (Optional) paste <strong>LRC</strong> or plain lyrics. If you leave it empty and check “Auto-generate,” the server will transcribe vocals with Whisper.</p>

    <form id="f" enctype="multipart/form-data" style="display:grid;gap:0.75rem">
      <input type="file" name="file" accept=".wav" required />
      <label style="display:flex;gap:.5rem;align-items:center;opacity:.9">
        <input type="checkbox" id="auto" checked />
        Auto-generate lyrics with Whisper
      </label>
      <textarea name="lyrics_lrc" id="lrc" rows="8"
        placeholder="[00:12.50] First line of lyrics
[00:18.20] Next line of lyrics"
        style="width:100%;min-height:8rem;padding:.75rem;border-radius:8px;border:1px solid #23304b;background:#0e1422;color:#cfe4ff;font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"></textarea>
      <div style="display:flex;gap:.5rem">
        <button>Analyze + Align</button>
        <button type="button" id="smk">Run built-in smoketest</button>
      </div>
    </form>

    <h2 style="margin-top:2rem">Chord-over-Lyrics</h2>
    <pre id="tab" style="background:#0e1422;padding:1rem;border-radius:8px;white-space:pre-wrap;min-height:8rem">–</pre>
  </div>

<script>
const f = document.getElementById('f');
const tab = document.getElementById('tab');
const auto = document.getElementById('auto');
const lrc = document.getElementById('lrc');

f.addEventListener('submit', async (e)=>{
  e.preventDefault();
  tab.textContent = 'Uploading…';

  const file = f.file.files[0];
  if (!file) { tab.textContent = 'Choose a WAV first.'; return; }

  // 1) Get upload URL
  const r1 = await fetch('/upload-url', {method:'POST'});
  const j1 = await r1.json();
  if (!j1.ok) { tab.textContent = 'Failed to get upload URL'; return; }

  // 2) Upload directly to GCS (resumable)
  const put = await fetch(j1.upload_url, {
    method: 'PUT',
    headers: { 'Content-Type': 'audio/wav', 'x-goog-resumable': 'start' },
    body: file
  });
  if (!put.ok) {
    tab.textContent = 'Upload failed: ' + put.status + ' ' + put.statusText;
    return;
  }

  tab.textContent = 'Analyzing…';

  // 3) Tell backend to analyze the uploaded object
  const r2 = await fetch('/analyze_gcs', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      bucket: j1.bucket,
      object: j1.object,
      lyrics_auto: auto.checked ? 'true' : 'false',
      lyrics_lrc: lrc.value || ''
    })
  });
  const j2 = await r2.json();
  if (!r2.ok || !j2.ok) {
    tab.textContent = (j2 && j2.error) ? j2.error : ('Analyze error ' + r2.status);
    return;
  }

  if (j2.aligned?.length) tab.textContent = j2.aligned.map(x => x.mono).join("\n\n");
  else if (j2.lyrics_lrc) tab.textContent = "Lyrics:\n\n" + j2.lyrics_lrc;
  else tab.textContent = "No lyrics provided/available.";
});
</script>
</body>
"""
    return HTMLResponse(html)

@app.get("/__whoami")
def whoami():
    return {
        "version": VERSION,
        "sonic": SONIC,
        "vamp_path": VAMP_PATH,
        "disable_sonic": DISABLE_SONIC,
        "platform": os.name,
        "cwd": str(pathlib.Path().resolve()),
    }

@app.get("/health")
def health():
    # Try listing transforms; if it fails, we still return something helpful.
    env = os.environ.copy(); env["VAMP_PATH"] = VAMP_PATH
    try:
        out = subprocess.check_output([SONIC, "-l"], env=env, stderr=subprocess.STDOUT, text=True, errors="ignore")
        has_chordino = any("chordino" in ln.lower() for ln in out.splitlines())
        return {"ok": True, "has_chordino": bool(has_chordino), "note": "look for 'chordino' in plugins list"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/_smoketest")
def smoketest():
    wav = synth_c_major_wav()
    try:
        try:
            lab = run_sonic_simplechord(wav)
            segs = merge_segments(parse_lab(lab), min_dur=0.4)
            labels = sorted(set(s["chord"] for s in segs))
            return {"ok": True, "labels": labels, "segments": segs, "raw_first_lines": lab.splitlines()[:8]}
        except Exception as chord_err:
            return {"ok": True, "note": f"Chord detector unavailable: {chord_err}", "segments": []}
    finally:
        try: os.remove(wav)
        except: pass


from fastapi import Form
from fastapi.responses import JSONResponse

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    lyrics_lrc: str | None = Form(default=None),
    lyrics_auto: str | None = Form(default=None),  # "true" to enable auto if lyrics missing
):
    wav = None
    try:
        # 1) Save upload (requires python-multipart in requirements.txt)
        raw = await file.read()
        wav = write_wav_from_bytes(raw)  # still enforces WAV for now

        # 2) Chord detection — NEVER fail the request if Sonic is off/missing
        segments_error = None
        final, pattern, sections = [], [], []
        try:
            lab = run_sonic_simplechord(wav)  # may raise when DISABLE_SONIC=1 or binary missing
            if lab and lab.strip():
                raw_segments = parse_lab(lab)
                if raw_segments:
                    final, pattern, sections = postprocess_variable_pattern(raw_segments)
        except Exception as e:
            # Try librosa fallback instead of failing
            try:
                final = fallback_chords_librosa(wav, hop_s = HOP)
                if final:
                    try:
                        tonic_pc, mode = estimate_key_krumhansl(wav)
                        if tonic_pc is not None:
                            final = key_aware_correct(
                                final, tonic_pc, mode,
                                drop_out_of_scale_if_short = True, short_thresh = 1.2
                            )
                    except Exception:
                        pass
            except Exception as e2:
                segments_error = f"Chord detection unavailable: {type(e).__name__}: {e} | Fallback failed: {type(e2).__name__}: {e2}"
            else:
                segments_error = "Chordino disabled/missing; used librosa fallback triads."
            # final/pattern/sections stay empty; continue

        # 3) Key detection + diatonic snap (only if we actually have segments)
        det_key = None
        if final:
            try:
                tonic_pc, mode = estimate_key_krumhansl(wav)
                if tonic_pc is not None:
                    det_key = f"{PC_NAMES[tonic_pc]} {'major' if mode == 'maj' else 'minor'}"
                    final = key_aware_correct(
                        final, tonic_pc, mode,
                        drop_out_of_scale_if_short=True, short_thresh=1.2
                    )
            except Exception:
                pass  # key detection is best-effort

        # 4) Lyrics: provided vs auto (Whisper)
        lrc_text = (lyrics_lrc or "").strip()
        asr_error = None
        if not lrc_text and (lyrics_auto or "").lower() == "true":
            try:
                lrc_text = whisper_to_lrc(wav, per_word=False)  # segment-level lines are cleaner for singing
            except Exception as asr_err:
                asr_error = f"Auto-lyrics failed: {type(asr_err).__name__}: {asr_err}"
                lrc_text = ""

        # 5) Align chords → words (works even if `final` is empty; you’ll still see lyrics)
        aligned = []
        if lrc_text:
            lines = parse_lrc(lrc_text)

            # If no timestamps at all, spread lines across song duration (or roughly equal spacing)
            if not any(ln.get("start") is not None for ln in lines):
                # try to get duration
                total = float(final[-1]["end"]) if final else 0.0
                if total <= 0 and librosa is not None:
                    try:
                        y, sr = librosa.load(wav, sr=None, mono=True)
                        total = len(y) / float(sr)
                    except Exception:
                        total = 0.0
                n = max(1, len(lines))
                step = (total or n) / n
                for i, ln in enumerate(lines):
                    ln["start"] = i * step
                    ln["end"] = (i + 1) * step if i < n - 1 else total

            # Fill missing ends from next line or song end
            song_end = float(final[-1]["end"]) if final else (
                lines[-1]["end"] if lines and lines[-1].get("end") is not None else None
            )
            for i, ln in enumerate(lines):
                if ln.get("start") is not None and ln.get("end") is None:
                    ln["end"] = lines[i + 1]["start"] if (i + 1 < len(lines) and lines[i + 1].get("start") is not None) else song_end

            aligned = align_chords_to_words(final, lines)

        # 6) Success response (HTTP 200 even if Sonic is disabled)
        return JSONResponse({
            "ok": True,
            "filename": file.filename,
            "detected_key": det_key,
            "detected_pattern": pattern,
            "sections": sections,
            "segments": final,            # [] when Sonic disabled or missing
            "segments_error": segments_error,  # human-readable note
            "lyrics_lrc": lrc_text or None,
            "aligned": aligned,           # each has .mono = "CHORDS\nLYRICS"
            "asr_error": asr_error,       # Whisper problems, if any
        })

    except Exception as e:
        # Only truly bad requests (e.g., not a WAV) land here
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            if wav and os.path.exists(wav):
                os.remove(wav)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
