import os
import argparse
from pathlib import Path
from typing import List

import torch
import soundfile as sf
from pydub import AudioSegment
from nemo.collections.asr.models import SortformerEncLabelModel

def diar_to_rttm_lines(recording_id: str, diar_lines: List[str]) -> List[str]:
    """Helper to convert NeMo diarization output lines to RTTM format."""
    out: List[str] = []
    for line in diar_lines:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        start = float(parts[0])
        end = float(parts[1])
        speaker = parts[2]
        duration = max(0.0, end - start)
        if duration <= 0:
            continue
        out.append(
            f"SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
    return out

def split_audio_and_rttm_by_speaker(audio_path: str, rttm_lines: List[str], output_dir: Path):
    """Splits the input audio and RTTM into separate files for each speaker."""
    print(f"Loading audio for splitting: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    rec_id = Path(audio_path).stem
    
    # Identify unique speakers and their segments
    speaker_data = {}
    for line in rttm_lines:
        parts = line.split()
        if len(parts) < 8:
            continue
        start_sec = float(parts[3])
        duration_sec = float(parts[4])
        speaker_id = parts[7]
        
        if speaker_id not in speaker_data:
            speaker_data[speaker_id] = {"segments": [], "rttm_lines": []}
        
        speaker_data[speaker_id]["segments"].append((start_sec, start_sec + duration_sec))
        speaker_data[speaker_id]["rttm_lines"].append(line)
    
    # Process each speaker
    for speaker_id, data in speaker_data.items():
        print(f"Exporting audio and RTTM for {speaker_id}...")
        
        # Split Audio
        speaker_audio = AudioSegment.empty()
        for start, end in data["segments"]:
            segment_audio = audio[int(start * 1000):int(end * 1000)]
            speaker_audio += segment_audio
        
        audio_out = output_dir / f"{rec_id}_{speaker_id}.wav"
        speaker_audio.export(str(audio_out), format="wav")
        
        # Split RTTM
        rttm_out = output_dir / f"{rec_id}_{speaker_id}.rttm"
        rttm_out.write_text("\n".join(data["rttm_lines"]) + "\n", encoding="utf-8")
        
        print(f"Saved: {audio_out} and {rttm_out}")

def main():
    parser = argparse.ArgumentParser(description="Diarize and split audio by speaker.")
    parser.add_argument("input", type=str, help="Path to input audio file")
    
    # Default outputs to 'output' folder
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--model", type=str, default="models/speaker_diarization/sortformer_streaming_4spk_v2.nemo", help="Path to the trained Sortformer model (.nemo file)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    model_path = Path(args.model)
    
    out_dir = Path(args.out_dir) / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        print("Please download the pre-trained model first by running:")
        print("    python download_model.py")
        return
        
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model: {model_path}")
    model = SortformerEncLabelModel.restore_from(str(model_path))
    model.eval()
    if device == "cuda":
        model = model.cuda()
    
    print(f"Running diarization on: {input_path}")
    diar_outputs = model.diarize(
        audio=[str(input_path)],
        batch_size=1,
        verbose=False,
    )
    
    diar_lines = diar_outputs[0] if diar_outputs else []
    rttm_lines = diar_to_rttm_lines(input_path.stem, diar_lines)
    
    # Save RTTM
    rttm_file = out_dir / f"{input_path.stem}.rttm"
    rttm_file.write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")
    print(f"RTTM saved to: {rttm_file}")
    
    # Split audio and RTTM
    if rttm_lines:
        split_audio_and_rttm_by_speaker(str(input_path), rttm_lines, out_dir)
    else:
        print("No speakers detected.")

if __name__ == "__main__":
    main()
