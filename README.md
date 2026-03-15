# Streaming Speaker Diarization with Sortformer

This project implements fine-tuning for the **Sortformer** model from NVIDIA (a Transformer-based encoder-labeler architecture) for the Speaker Diarization task ("who spoke when"). 

The model is optimized for streaming audio and is capable of separating the voices of up to **4 simultaneous speakers**, maintaining high accuracy even in scenarios with overlapping speech.

---

## Key Features
*   **Impressive Accuracy**: Achieved an F1 Score of 93.8% and reduced the Diarization Error Rate (DER) by 77% (down to 2.14%) compared to the base model.
*   **Powerful Base Model**: Utilizes the `nvidia/diar_streaming_sortformer_4spk-v2` base model (117 million parameters).
*   **Comprehensive Toolset**: Provides essential scripts from model training (`train.py`) to automated inference and audio splitting (`inference.py`).

---

## Environment Setup (NeMo Toolkit)

This project requires installing the core **NVIDIA NeMo Toolkit** from its GitHub source to ensure compatibility and full configuration for automatic speech recognition (ASR) pipelines.

```bash
# Clone the NeMo repository
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo

# Install the NeMo framework and ASR dependencies
pip install -e ".[asr]"
```
*(Note: It is required to have a PyTorch version installed that matches your system's CUDA version prior to this).*

---

## Model Training (`train.py`)

Run `train.py` to fine-tune the model based on specific configurations. The core of fine-tuning involves pointing the Manifest filepath for the Train and Val datasets into the PyTorch Lightning training system.

**Detailed Training Example Command:**
```bash
python train.py --exp-name sortformer_streaming_4spk_v2 --lr 1e-5 --max-epochs 12 --es-patience 3
```

*During training, the script automatically monitors `val_f1_acc` and saves the best model as a `.nemo` file in the `experiments/.../version_X/checkpoints/` directory using an Early Stopping mechanism.*

---

## Inference and Automated Audio Splitting (`inference.py`)

Use `inference.py` to run the model to identify speaker segments on any given audio file, which automates the extraction and splitting of individual speakers' voices independently. The path to the best model has been embedded as the default in the core configuration of the script.

**1. Simplified Inference Command:**
The script is pre-configured with the default parameters; you only need to provide the target `.wav` audio file:
```bash
python inference.py "dataset/testing_audio/meeting_4_people.wav"
```

**2. Output Structure:**
By design, the system will automatically generate a subdirectory named after the original audio file inside the `output/` directory:
```text
output/
└── meeting_4_people/
    ├── meeting_4_people_speaker_0.rttm  # RTTM logging individual timeline of Speaker 0
    ├── meeting_4_people_speaker_0.wav   # Concatenated voice streams exclusively for Speaker 0
    ├── meeting_4_people_speaker_1.rttm
    ├── meeting_4_people_speaker_1.wav
    ├── ...
```
This audio separation process completely removes empty structures (e.g., general silences) and provides extremely clean voice tracks ready for downstream NLP processing.
