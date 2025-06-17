# EDA - 통계 파트 코드
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram

'''주어진 세 개의 .wav 파일(spk1, spk2, mix)을 읽고,
분리 관련 주요 통계값을 계산하여 dict로 반환.'''
def analyze_wav_pair(spk1_path, spk2_path, mix_path):
    try:
        sr1, y1 = wavfile.read(spk1_path)
        sr2, y2 = wavfile.read(spk2_path)
        sr3, y3 = wavfile.read(mix_path)

        y1, y2, y3 = [x.astype(np.float32) if x.ndim == 1 else x[:, 0].astype(np.float32)
                     for x in [y1, y2, y3]]

        duration = len(y3) / sr3
        rms1 = np.sqrt(np.mean(y1**2))
        rms2 = np.sqrt(np.mean(y2**2))
        rms_mix = np.sqrt(np.mean(y3**2))

        def center_freq(y, sr):
            f, _, Sxx = spectrogram(y, fs=sr, nperseg=512)
            Sxx_mean = np.mean(Sxx, axis=1)
            return np.sum(f * Sxx_mean) / np.sum(Sxx_mean)

        cf1 = center_freq(y1, sr1)
        cf2 = center_freq(y2, sr2)
        cf_mix = center_freq(y3, sr3)

        f, t, Sxx1 = spectrogram(y1, fs=sr1)
        _, _, Sxx2 = spectrogram(y2, fs=sr2)
        min_len = min(Sxx1.shape[1], Sxx2.shape[1])
        stft_corr = np.corrcoef(Sxx1[:, :min_len].flatten(), Sxx2[:, :min_len].flatten())[0, 1]

        return {
            "filename": os.path.basename(mix_path),
            "rms_spk1": rms1,
            "rms_spk2": rms2,
            "rms_mix": rms_mix,
            "cf_spk1": cf1,
            "cf_spk2": cf2,
            "cf_mix": cf_mix,
            "cf_diff": abs(cf1 - cf2),
            "stft_corr": stft_corr
        }

    except Exception as e:
        return {"filename": os.path.basename(mix_path), "error": str(e)}

# 전체 subset 반복
base_path = "/content/drive/MyDrive/final_data" #참조: https://drive.google.com/file/d/1odQm9jrT03vR3z78yDJt0k169AYluz3M/view?usp=sharing
subsets = ["train", "val", "test"]
results = []

for subset in subsets:
    spk1_dir = os.path.join(base_path, subset, "spk1")
    spk2_dir = os.path.join(base_path, subset, "spk2")
    mix_dir  = os.path.join(base_path, subset, "mixtures")

    for fname in sorted(os.listdir(mix_dir)):
        if not fname.endswith(".wav"):
            continue
        idx = fname.split("_")[-1]
        spk1_file = os.path.join(spk1_dir, f"spk1_reverb_{idx}")
        spk2_file = os.path.join(spk2_dir, f"spk2_reverb_{idx}")
        mix_file  = os.path.join(mix_dir, fname)

        result = analyze_wav_pair(spk1_file, spk2_file, mix_file)
        result["subset"] = subset
        results.append(result)

# 저장
df = pd.DataFrame(results)
df.to_csv("/content/drive/MyDrive/TargetedANC/Data_EDA/final_audio_eda_report.csv", index=False)
df.head()