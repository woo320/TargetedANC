#!/usr/bin/env python3
"""Joint model inference + audio & spectrogram visualisation
5‑track 오디오(라벨) + 2‑way 스펙트로그램(Mix / Final)
- 첫 skip_first_n 파일은 평균 속도 계산에서 제외하되,
  시각화 샘플은 워밍업 여부와 상관없이 최대 visualize_n 개까지 수집.
"""
import os
import time
import glob
import traceback

import torch
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display

from models.joint_model import ImprovedJointModel
from project_utils.audio_utils import standardize_audio_dims
from config.constants import SR
from loss_function import dBA_Loss

# ───────────────────────────────────────────────────────────────────
# 시각화 함수
# ───────────────────────────────────────────────────────────────────

def visualize_results(samples, sr, avg_ms_per_s):
    print(f"Average processing speed (after warm‑up): {avg_ms_per_s:.1f} ms/s (RTF≈{avg_ms_per_s/1000:.3f})")

    labels = [
        ("mix", "Mixture (mic)"),
        ("spk1", "s1 / Broadcast"),
        ("spk2", "s2 / Noise"),
        ("anti", "Anti‑noise"),
        ("output", "s1 + denoised (Final)")
    ]

    for idx, s in enumerate(samples, 1):
        print(f"===== Sample {idx} =====")
        for key, text in labels:
            print(text)
            display(Audio(s[key], rate=sr))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, (title, wav) in zip(axes, [("Mix", s["mix"]), ("Final", s["output"]) ]):
            S = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
            librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz", ax=ax)
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

# ───────────────────────────────────────────────────────────────────
# 모델 로드 (생략: 이전과 동일)
# ───────────────────────────────────────────────────────────────────

def load_joint_model(joint_checkpoint_path, device):
    ckpt = torch.load(joint_checkpoint_path, map_location=device, weights_only=False)
    if "config" in ckpt:
        cfg = ckpt["config"]
        sud_ckpt = cfg["sudormrf_checkpoint"]
        wav_ckpt = cfg["wavenet_checkpoint"]
        wav_cfg  = cfg["wavenet_config"]
        cls_ckpt = cfg.get("broadcast_classifier_checkpoint")
        use_cls  = cfg.get("use_broadcast_classifier", True)
    else:
        sud_ckpt = "/Joint/weight/separate.pt"
        wav_ckpt = "/Joint/weight/reduction.pth"
        wav_cfg  = "/Joint/weight/config_opt_210.json"
        cls_ckpt = "/Joint/weight/classifier.pth"
        use_cls  = True

    model = ImprovedJointModel(
        sudormrf_checkpoint_path=sud_ckpt,
        wavenet_checkpoint_path=wav_ckpt,
        wavenet_config_path=wav_cfg,
        broadcast_classifier_checkpoint_path=cls_ckpt,
        use_broadcast_classifier=use_cls,
    ).to(device)

    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    if any(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# ───────────────────────────────────────────────────────────────────
# 추론 함수
# ───────────────────────────────────────────────────────────────────

def run_joint_inference(
    joint_ckpt,
    input_dir,
    sep_out,
    noise_out,
    anti_out,
    denoise_out,
    final_out,
    *,
    visualize_n=5,
    use_inference_style=True,
    save_debug=False,
    skip_first_n=3,
):
    for d in [sep_out, noise_out, anti_out, denoise_out, final_out]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_joint_model(joint_ckpt, device)

    files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if not files:
        print("No wav files found.")
        return

    speed_msps, vis = [], []

    for idx, p in enumerate(files):
        try:
            wav, fs = sf.read(p)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if fs != SR:
                wav = librosa.resample(wav, fs, SR)

            t0 = time.time()
            x = standardize_audio_dims(torch.from_numpy(wav).float().to(device))
            with torch.no_grad():
                outs = (model.forward_inference_style if use_inference_style else model.forward_for_training)(
                    x, chunk_len=int(4*SR), return_classification=True)
            t1 = time.time()

            spk1 = outs["s1_clean"].squeeze().cpu().numpy()
            spk2 = outs["s2_noise"].squeeze().cpu().numpy()
            anti = outs["s2_antinoise"].squeeze().cpu().numpy()
            out  = outs["enhanced_verification"].squeeze().cpu().numpy()

            # ⓐ 평균 속도: 워밍업 이후만
            if idx >= skip_first_n:
                dur_s = len(wav)/SR
                speed_msps.append((t1 - t0) * 1000 / dur_s)

            # ⓑ 시각화: 워밍업 여부 무관, 최대 visualize_n 개
            if len(vis) < visualize_n:
                vis.append({
                    "mix": wav,
                    "spk1": spk1,
                    "spk2": spk2,
                    "anti": anti,
                    "output": out,
                })

            # 결과 파일 저장
            sf.write(os.path.join(final_out, f"{idx:03d}_final.wav"), out, SR)
            if save_debug:
                sf.write(os.path.join(denoise_out, f"{idx:03d}_anti.wav"), anti, SR)

        except Exception:
            traceback.print_exc()
            continue

    if speed_msps:
        visualize_results(vis, SR, sum(speed_msps) / len(speed_msps))

# ───────────────────────────────────────────────────────────────────
# 래퍼
# ───────────────────────────────────────────────────────────────────

def quick_inference(joint_model_path, input_audio_dir, output_base_dir, skip_first=3):
    out_dirs = {
        k: os.path.join(output_base_dir, v)
        for k, v in {
            "sep_out": "broadcast",
            "noise_out": "noise",
            "anti_out": "antinoise",
            "denoise_out": "denoised",
            "final_out": "final",
        }.items()
    }

    run_joint_inference(
        joint_ckpt=joint_model_path,
        input_dir=input_audio_dir,
        visualize_n=5,
        use_inference_style=True,
        save_debug=False,
        skip_first_n=skip_first,
        **out_dirs,
    )

# ───────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    JOINT_MODEL_PATH = "/content/drive/MyDrive/joint/result/joint_mixed_training_2025-06-13-15h04m/weights/best_composite.pth"
    INPUT_DIR        = "/content/drive/MyDrive/TargetedANC/inference_testdata"
    OUTPUT_DIR       = "/content/drive/MyDrive/inference_outputs"

    quick_inference(JOINT_MODEL_PATH, INPUT_DIR, OUTPUT_DIR)
