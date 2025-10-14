# Usage:
# Single video:
#   python preprocess.py --url ... --audio_out Master/1/Audio/audio.wav --transcript_out Master/1/Transcript/transcript.txt


import argparse
import json, csv
import shutil
import subprocess
from pathlib import Path
import sys, os
import torch
from typing import Dict, List, Optional
import whisper
import re

COOKIES_FILE = "youtube_cookies.txt"
WHISPER_MODEL_PATH = "/WAVE/projects/oignat_lab/Mentorship-VQA/whisper-model/large-v3.pt"

_SENT_FINAL = set(".?!…？！。") | {"।"}
_CONTINUATION = set(",;:，；：")

# ---------- IO helpers ----------

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download_audio(url: str, audio_out: Path) -> Path:
    """
    Downloads the best quality audio from a YouTube URL as a WAV file.
    Returns the path to the downloaded audio file.
    """
    
    # remove the audio folder
    if os.path.exists(audio_out):
        shutil.rmtree(audio_out)
        print(f"🗑️  Removed existing '{audio_out}' folder.")
        
    os.makedirs(audio_out, exist_ok=True)
    
    audio_path = os.path.join(audio_out, "audio.wav")
    print(f"--- Downloading Audio from: {url} ---")
    cmd = [
        "yt-dlp",
        '--cookies', COOKIES_FILE,
        '--no-playlist',
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "-o", audio_path,
        url,
    ]

    print("⏳ downloading audio:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not audio_out.exists():
        raise RuntimeError(f"audio expected at {audio_out} but not found")
    print(f"✅ audio saved to {audio_out}")
    return audio_path

# ---------- Language helpers ----------

# def _load_hindi_punc_model():
#     """Load and return the Hindi punctuation model."""
#     global hindi_punc_model
#     if hindi_punc_model is None:
#         print("    -> Loading Hindi Punctuation Model...")
#         # Replace 'hi' with the correct model identifier if necessary.
#         # This is where you would initialize your specific model.
#         # Example for a multilingual model:
#         hindi_punc_model = PunctuationModel(model="ai4bharat/Cadence", cpu = False)
#         print("    -> Hindi Punctuation Model loaded.")
#     return hindi_punc_model

def _choose_end_mark(lang_code: str, text: str) -> str:
    # Prefer language-specific sentence enders where appropriate.
    if lang_code == "hi":
        return "।"
    if lang_code in ("zh", "zh-hans", "zh-hant", "ja", "ko"):
        return "。"
    # default .
    return "."

def _already_ended(s: str) -> bool:
    s = s.rstrip()
    if not s:
        return True
    # If ends with ellipsis '...' or '…', treat as ended
    if re.search(r'(\.\.\.|…)$', s):
        return True
    last = s[-1]
    # Skip if last char is sentence-final OR continuation punctuation
    return last in _SENT_FINAL or last in _CONTINUATION

def restore_punctuation_text(text: str, lang_code: str) -> str:
    """
    Restore punctuation on a piece of text using deepmultilingualpunctuation.
    Optionally map period to danda for Hindi.
    """
    if not text:
        return text
    t = re.sub(r"\s+", " ", text.strip())
    # if lang_code == "hi":
    #     try:
    #         model = _load_hindi_punc_model()
    #         # Use the model's restore method
    #         restored_text = model.restore_punctuation(t)
    #         return restored_text
    #     except Exception as e:
    #         # Fallback to simple period/danda addition if model fails
    #         print(f"Warning: Hindi Punctuation Model failed ({e}). Falling back to simple punctuation.")
    #         pass 
    
    if _already_ended(t):
        return t
    mark = _choose_end_mark(lang_code, t)
    return t + mark 

def transcribe(audio_path: Path, transcript_out: Path, lang: str):
    """
    Transcribes audio and translates it to English using the local Whisper model.
    Saves the transcript to a text file.
    """
    ensure_parent(transcript_out)
    # Check for GPU and load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Using device: {device}")
    
    print(f"⏳ loading whisper model: {WHISPER_MODEL_PATH}")

    model = whisper.load_model(WHISPER_MODEL_PATH, device=device)
    
    # Language detection with Whisper detection
    
    # audio = whisper.load_audio(str(audio_path))
    # audio = whisper.pad_or_trim(audio)  # trims/pads to 30s for detection
    # mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
    # _, lang_probs = model.detect_language(mel)
    # detected_lang = max(lang_probs, key=lang_probs.get)
    # print(f"Language is {detected_lang} with prob {lang_probs[detected_lang]:.3f}")
    
    # --- Map language code to filename suffix ---
    lang_map = {
        "chinese": "zh",
        "hindi": "hi",
        "romanian": "ro",
        "english": "en"
    }
    suffix = lang_map.get(lang.lower(), lang.lower())
    
    prompts = {
    "zh": "请使用标点符号。",
    "hi": "कृपया हिंदी में लिखें और '।' जैसे विराम चिह्नों का उपयोग करें",
    "ro": "Transcrie în limba română și folosește semne de punctuație.",
    "en": "Transcribe in English and use punctuation.",
    }
    
    init_prompt = prompts.get(suffix)
    if not init_prompt:  
        init_prompt = "Please use punctuation."
    print(f"🔎 Using initial prompt: {repr(init_prompt)}")
    
    print("   -> Starting transcription (this may take a while)...")
    # Note: if the detected language is Hindi, we don't use initial prompt
    if suffix == "hi":
        result = model.transcribe(
            str(audio_path), 
            task = "transcribe")
    else:
        result = model.transcribe(
            str(audio_path),
            initial_prompt= init_prompt,
            task ="transcribe" 
        )
    
    # detected_lang = result.get("language", "unknown")
    print(f"🌐 Using language from argument: {suffix}")

    # Apply punctuation restoration to each segment
    # Restore punctuation for Chinese only
    print("   -> Restoring punctuation on segments...")
    if suffix in ("zh"):
        for seg in result.get("segments", []):
            seg_text = seg.get("text", "")
            seg["text"] = restore_punctuation_text(seg_text, suffix)

    
    # --- 1. Save the structured JSON output ---
    json_filename = f"transcript-{suffix}.json"
    json_path = transcript_out / json_filename
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2, ensure_ascii=False)
    print(f"✅ Raw segment data saved to: {json_path}")

    # --- 2. Create and save the SRT subtitle file ---
    srt_path = os.path.join(transcript_out, f"transcript_{suffix}.srt")
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result['segments'], 1):
            # Format start and end times to SRT format (HH:MM:SS,ms)
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            
            # Write the segment number, timestamp, and text to the file
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{segment['text'].strip()}\n\n")

    print(f"✅ Subtitle file saved to: {srt_path}")
    print("-" * 30)

def format_timestamp(seconds):
    """Converts seconds to SRT time format HH:MM:SS,ms."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# ---------- Spreadsheet reading ----------

def read_csv(path: Path) -> List[Dict[str, str]]:
    """
    Expect columns: url (required), index (optional), title (optional).
    Returns sorted list by 'index' (auto-numbered if missing).
    """
    rows: List[Dict[str, str]] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k.strip(): (v or "").strip() for k, v in r.items()})
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        try:
            import pandas as pd  # pandas + openpyxl recommended for xlsx
        except ImportError as e:
            raise SystemExit("Please `pip install pandas openpyxl` to read .xlsx files") from e
        df = pd.read_excel(path, dtype=str).fillna("")
        rows = df.to_dict(orient="records")
    else:
        raise ValueError("Unsupported sheet format. Use .csv or .xlsx")

    # normalize + auto index
    out: List[Dict[str, str]] = []
    autoinc = 1
    for r in rows:
        url = r.get("url", "").strip()
        if not url:
            continue  # skip blank rows
        idx_raw = r.get("index", "").strip()
        if idx_raw == "":
            idx = autoinc
            autoinc += 1
        else:
            try:
                idx = int(idx_raw)
            except ValueError:
                idx = autoinc
                autoinc += 1
        language = r.get("language", "").strip()
        out.append({"index": idx, "url": url, "language": language})
    out.sort(key=lambda x: x["index"])
    return out

# ---------- Batch driver ----------

def process_single(
    url: str,
    audio_out: Path,
    transcript_out: Path,
    lang: str,
    cookies: Optional[str],
    dry_run: bool = False,
):
    print(f"🎬 processing {url}")
    if dry_run:
        print(f"DRY-RUN would save audio → {audio_out}")
        print(f"DRY-RUN would save transcript → {transcript_out}")
        return
    audio = download_audio(url, audio_out)
    transcribe(audio, transcript_out, lang)

def process_sheet(
    sheet_path: Path,
    base_dir: Path,
    only: Optional[List[int]],
    cookies: Optional[str],
    dry_run: bool = False,
):
    videos = read_csv(sheet_path)
    if only:
        only_set = set(only)
        videos = [v for v in videos if v["index"] in only_set]

    safe_mkdir(base_dir)

    for v in videos:
        idx, url, language = v["index"], v["url"], v["language"]
        print(f"\n===== Video {idx}: {language} =====")
        root = base_dir / f"{idx}"
        audio_dir = root / "Audio"
        transcript_dir = root / "Transcript"
        safe_mkdir(audio_dir)
        safe_mkdir(transcript_dir)

        process_single(
            url=url,
            audio_out=audio_dir,            # pass the folder to save audio
            transcript_out=transcript_dir,  
            cookies=cookies,
            dry_run=dry_run,
            lang = language
        )

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Download audio (yt-dlp) and transcribe (Whisper) from either a single URL or a spreadsheet."
    )
    # mode: single
    ap.add_argument("--url", help="YouTube URL (single mode)")
    ap.add_argument("--audio_out", help="Where to write audio file (single mode)")
    ap.add_argument("--transcript_out", help="Where to write transcript (single mode)")
    ap.add_argument("--lang", help="The language of this video")
    
    # mode B: spreadsheet
    ap.add_argument("--sheet", help="Path to CSV or XLSX with columns: index,url,[title]")
    ap.add_argument("--base", default="OUTPUT", help="Base output folder for sheet mode (default: OUTPUT)")
    ap.add_argument("--only", nargs="*", type=int, help="Only process these indices (sheet mode)")
    
    # shared options
    # could remove this argument
    ap.add_argument("--cookies", default=None, help="Path to youtube cookies.txt (optional)")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan without doing work")
    args = ap.parse_args()

    # Validate mode
    single_mode = args.url is not None
    sheet_mode = args.sheet is not None
    if not single_mode and not sheet_mode:
        print("❌ Single: --url --audio_out --transcript_out \n Sheet: --url --base", file=sys.stderr)
        sys.exit(2)

    try:
        if single_mode:
            if not args.audio_out or not args.transcript_out or not args.lang:
                print("❌ Requires --audio_out, --lang and --transcript_out", file=sys.stderr)
                sys.exit(2)
            process_single(
                url=args.url,
                audio_out=Path(args.audio_out),
                transcript_out=Path(args.transcript_out),
                cookies=args.cookies,
                dry_run=args.dry_run,
                lang=args.lang
            )
        else:
            process_sheet(
                sheet_path=Path(args.sheet),
                base_dir=Path(args.base),
                only=args.only,
                cookies=args.cookies,
                dry_run=args.dry_run,
            )
        print("\n🎉 done.")
    except subprocess.CalledProcessError as e:
        print("❌ external tool failed:", e, file=sys.stderr)
        sys.exit(e.returncode or 1)
    except Exception as e:
        print("❌ error:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
