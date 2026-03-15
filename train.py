import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeMo streaming sortformer diarizer")
    parser.add_argument("--train-manifest", default="dataset/manifests/train.json")
    parser.add_argument("--val-manifest", default="dataset/manifests/val.json")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=None,
        help="Use only a fraction (0,1] or fixed number (>1) of train batches per epoch",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Use only a fraction (0,1] or fixed number (>1) of val batches",
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=None,
        help="Run validation every N epochs (optional, NeMo config append)",
    )
    parser.add_argument("--exp-dir", default="experiments/sortformer")
    parser.add_argument("--exp-name", default="sortformer_streaming_4spk")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints in exp_dir/exp_name",
    )
    parser.add_argument("--pretrained-model", default="nvidia/diar_streaming_sortformer_4spk-v2")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable loading pretrained checkpoint and train from scratch",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Force conservative memory settings for small GPUs",
    )
    parser.add_argument(
        "--session-len-sec",
        type=int,
        default=None,
        help="Override model.train_ds/validation_ds.session_len_sec",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping callback (enabled by default)",
    )
    parser.add_argument(
        "--es-monitor",
        default="val_f1_acc",
        help="Metric to monitor for early stopping",
    )
    parser.add_argument(
        "--es-mode",
        choices=["min", "max"],
        default="max",
        help="Early stopping mode",
    )
    parser.add_argument(
        "--es-patience",
        type=int,
        default=3,
        help="Number of validation checks with no improvement before stop",
    )
    parser.add_argument(
        "--es-min-delta",
        type=float,
        default=0.001,
        help="Minimum change to qualify as an improvement",
    )
    return parser.parse_args()


def check_manifest_max_speakers(manifest_path: Path, allowed_max_speakers: int = 4) -> None:
    with manifest_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            nspk = data.get("num_speakers")
            if isinstance(nspk, int) and nspk > allowed_max_speakers:
                raise ValueError(
                    f"{manifest_path} has num_speakers={nspk} at line {idx}. "
                    f"This script is 4spk-only. Please use a <=4 speaker manifest."
                )


def detect_low_vram() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total_gb <= 6.0
    except Exception:
        return False


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    nemo_root = project_root / "NeMo"

    train_manifest = project_root / args.train_manifest
    val_manifest = project_root / args.val_manifest
    exp_dir = project_root / args.exp_dir

    train_script = nemo_root / "examples" / "speaker_tasks" / "diarization" / "neural_diarizer" / "streaming_sortformer_diar_train.py"
    config_path = nemo_root / "examples" / "speaker_tasks" / "diarization" / "conf" / "neural_diarizer"
    config_name = "streaming_sortformer_diarizer_4spk-v2.yaml"

    required_paths = [
        train_script,
        config_path / config_name,
        train_manifest,
        val_manifest,
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"- {p}")
        sys.exit(1)

    check_manifest_max_speakers(train_manifest, allowed_max_speakers=4)
    check_manifest_max_speakers(val_manifest, allowed_max_speakers=4)

    exp_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    accelerator = "gpu" if has_cuda else "cpu"
    precision = "16" if has_cuda else "32"

    use_pretrained = not args.no_pretrained

    auto_low_vram = detect_low_vram()
    low_vram_mode = args.low_vram or auto_low_vram

    batch_size = args.batch_size
    session_len_sec = args.session_len_sec
    if low_vram_mode:
        if batch_size > 1:
            batch_size = 1
        if session_len_sec is None:
            session_len_sec = 30
        print("Low-VRAM mode enabled: batch_size=1, session_len_sec=30, grad_accum=2")

    num_workers = args.num_workers
    if os.name == "nt" and num_workers > 0:
        print(
            "Windows detected: forcing num_workers=0 to avoid multiprocessing "
            "pickling errors with NeMo datasets."
        )
        num_workers = 0

    cmd = [
        sys.executable,
        str(train_script),
        f"--config-path={config_path}",
        f"--config-name={config_name}",
        f"model.train_ds.manifest_filepath={train_manifest}",
        f"model.validation_ds.manifest_filepath={val_manifest}",
        "trainer.devices=1",
        f"trainer.accelerator={accelerator}",
        "trainer.strategy=auto",
        f"trainer.max_epochs={args.max_epochs}",
        "trainer.val_check_interval=1.0",
        f"trainer.precision={precision}",
        f"model.lr={args.lr}",
        f"model.optim.lr={args.lr}",
        "+trainer.enable_progress_bar=True",
        f"model.train_ds.num_workers={num_workers}",
        f"model.validation_ds.num_workers={num_workers}",
        f"exp_manager.exp_dir={exp_dir}",
        f"exp_manager.name={args.exp_name}",
        f"exp_manager.resume_if_exists={args.resume}",
        "model.max_num_of_spks=4",
        f"model.train_ds.batch_size={batch_size}",
        f"model.validation_ds.batch_size={batch_size}",
    ]

    if session_len_sec is not None:
        cmd += [
            f"model.train_ds.session_len_sec={session_len_sec}",
            f"model.validation_ds.session_len_sec={session_len_sec}",
        ]

    if args.limit_train_batches is not None:
        cmd.append(f"trainer.limit_train_batches={args.limit_train_batches}")
    if args.limit_val_batches is not None:
        cmd.append(f"trainer.limit_val_batches={args.limit_val_batches}")
    if args.check_val_every_n_epoch is not None:
        cmd.append(f"+trainer.check_val_every_n_epoch={args.check_val_every_n_epoch}")

    if low_vram_mode:
        cmd += [
            "trainer.accumulate_grad_batches=2",
            "model.train_ds.use_bucketing=False",
            "model.train_ds.max_duration=30",
            "model.train_ds.batch_duration=80",
            "model.train_ds.quadratic_duration=200",
        ]

    if not args.no_early_stop:
        cmd += [
            "++exp_manager.create_early_stopping_callback=True",
            f"++exp_manager.early_stopping_callback_params.monitor={args.es_monitor}",
            f"++exp_manager.early_stopping_callback_params.mode={args.es_mode}",
            f"++exp_manager.early_stopping_callback_params.patience={args.es_patience}",
            f"++exp_manager.early_stopping_callback_params.min_delta={args.es_min_delta}",
            "++exp_manager.early_stopping_callback_params.check_on_train_epoch_end=False",
            "++exp_manager.disable_validation_on_resume=False",
        ]

    if use_pretrained:
        cmd.append(f"+init_from_pretrained_model={args.pretrained_model}")

    run_env = os.environ.copy()
    run_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("Starting training with command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(project_root), env=run_env)


if __name__ == "__main__":
    main()
