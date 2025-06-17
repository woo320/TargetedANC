# TargetedANC



## Model Architecture
![image](https://github.com/user-attachments/assets/4b1b942b-d494-4a87-baa0-9376238e0d46)

## Setup
```
pip install -r requirements.txt
```

실행 환경: Pyhton: 3.11.13 / Torch: 2.6.0 / Torchaudio: 2.6.0 / librosa: 0.11.0 / Soundfile: 0.13.1

---

**Benchmark Dataset**

Demand: https://www.kaggle.com/datasets/chrisfilo/demand

ms_snsd: https://www.kaggle.com/datasets/jiangwq666/ms-snsd


## Training Data
Training Data Download Link : https://drive.google.com/file/d/1odQm9jrT03vR3z78yDJt0k169AYluz3M/view?usp=sharing
<div align="center">
<img src="https://github.com/user-attachments/assets/a02c34e8-2300-44c7-8ad5-4a4dd7f9cf3f" alt="Dataset Analysis Tables" width="600">
</div>

총 36시간의 학습 데이터셋

서울 교통 공사의 1~8호선 안내방송음과 AI HUB의 도시 소음 데이터를 사용

---

### Airplane Data
Airplane Data Download Link : https://drive.google.com/file/d/1sAq702S0YB-UkHnM5RuQGnYXmg5fniwf/view?usp=sharing

<div align="center">
<img src="https://github.com/user-attachments/assets/19263f40-3618-4bf6-b5b0-6e17ee62fd17" alt="airplane_data_statistic" width="600">
</div>

총 18시간의 항공 데이터셋

Simplaza의 74개의 항공사 기내 안전 안내방송 음원과 AI HUB의 도시 소음 데이터 중 비행기 소음 데이터를 사용

---

## Inference_Pipeline
**분리, 분류, 저감 각 모델의 Training과 Inference 코드이며, 아래의 내용은 Command line usage example입니다.**

**실행 환경은 Google Colab을 기준으로 작성되었습니다.**  
**모든 out 및 output 인자들은 빈 디렉토리로 미리 준비해두어야 합니다.**

### C-SuDoRM-RF++ Training Command
```bash
!python /content/drive/MyDrive/TargetedANC/inference_pipeline/C_SudoRM_RF/c_sudormrf_train.py \
  --model_type causal \
  --train "ANNOUNCENOISE" \
  --val "ANNOUNCENOISE" \
  --test "ANNOUNCENOISE" \
  --n_channels 1 \
  -fs 16000 \
  --batch_size 8 \
  --n_epochs 200 \
  --audio_timelength 4. \
  --enc_kernel_size 21 \
  --enc_num_basis 256 \
  --in_channels 512 \
  --out_channels 256 \
  --num_blocks 18 \
  -lr 0.001 \
  --divide_lr_by 3. \
  --patience 10 \
  --early_stop_patience 30 \
  --upsampling_depth 5 \
  --max_num_sources 2 \
  --min_num_sources 2 \
  --zero_pad_audio \
  --normalize_audio \
  -cad 0 \
  --n_jobs 4 \
  -clp <your_checkpoint_dir>
```




### C-SuDoRM-RF++ Inference Command
```bash
!python /content/drive/MyDrive/TargetedANC/inference_pipeline/C_SudoRM_RF/c_sudormrf_inference.py \
  -ckpt /content/drive/MyDrive/TargetedANC/inference_pipeline/C_SudoRM_RF/causal_best.pt \
  --input_dir /content/drive/MyDrive/TargetedANC/inference_testdata \
  --output_dir <your_output_dir>
```

---

### Audio Segment Classifier(ASC) Training Command
```bash
!python /content/drive/MyDrive/TargetedANC/inference_pipeline/ASC/ASC_train.py \
    --train_s1_dir /content/drive/MyDrive/final_data/train/spk1 \
    --train_s2_dir /content/drive/MyDrive/final_data/train/spk2 \
    --val_s1_dir     /content/drive/MyDrive/final_data/val/spk1 \
    --val_s2_dir     /content/drive/MyDrive/final_data/val/spk2 \
    --test_s1_dir    /content/drive/MyDrive/final_data/test/spk1 \
    --test_s2_dir    /content/drive/MyDrive/final_data/test/spk2 \
    --save_path <your_checkpoint_dir> \
    --sr 16000 \
    --window_len 16000 \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 15
```

### Audio Segment Classifier(ASC) Inference Command
```bash
!python /content/drive/MyDrive/TargetedANC/inference_pipeline/ASC/ASC_inference.py \
  --test_s1_dir <your_broadcast_dir> \
  --test_s2_dir <your_noise_dir> \
  --model_path /content/drive/MyDrive/inference_pipeline/ASC/asc.pth
```

---


### WaveNet-VNNs Training Command
```bash
!python /content/drive/MyDrive/TargetedANC/inference_pipeline/WaveNet_VNNs/train_opt_210.py \
  --config /content/drive/MyDrive/TargetedANC/inference_pipeline/WaveNet_VNNs/cfg_train_opt_210.toml \
  --device 0
```



### WaveNet-VNNs Infernce Command
```bash
!python /content/drive/MyDrive/TargetedANC/inference_pipeline/WaveNet_VNNs/inference_opt.py \
  --model-path /content/drive/MyDrive/TargetedANC/inference_pipeline/WaveNet_VNNs/model.pth \
  --config /content/drive/MyDrive/TargetedANC/inference_pipeline/WaveNet_VNNs/config_opt_210.json \
  --test-data-dir /content/drive/MyDrive/TargetedANC/inference_testdata \
  --output-enh-dir <your_denoise_dir> \
  --output-anti-dir <your_antinoise_dir>
```

---

### EndtoEnd Inference Command
```bash
!python /content/drive/MyDrive/TargetedANC/inference_pipeline/end2end_inference.py \
  --sep_ckpt      /content/drive/MyDrive/TargetedANC/inference_pipeline/C_SudoRM_RF/causal_best.pt \
  --noise_cfg     /content/drive/MyDrive/TargetedANC/inference_pipeline/WaveNet_VNNs/config_opt_210.json \
  --noise_ckpt   /content/drive/MyDrive/TargetedANC/inference_pipeline/WaveNet_VNNs/model.pth \
  --bcd_ckpt      /content/drive/MyDrive/TargetedANC/inference_pipeline/ASC/asc.pth \
  --input_dir     /content/drive/MyDrive/TargetedANC/inference_testdata \
  --sep_out      <your_seperation_dir> \
  --noise_out       <your_noise_dir> \
  --denoise_out      <your_denoise_dir> \
  --anti_out     <your_antinoise_dir> \
  --final_out   <your_final_dir>
```

## Joint Inference Examples

Joint/colab_Inference_Example.ipynb 를 통해 5개의 test data로 joint 추론 결과를 확인하실 수 있습니다.

Mix: 복합 소리 / Final: 분리 + 저감된 소음

![image](https://github.com/user-attachments/assets/6967239f-373c-4a39-a5f1-7bad2dfaafa9)


## References

C-SudoRMRF++:
https://github.com/etzinis/sudo_rm_rf

WaveNet-VNNs:
https://github.com/Lu-Baihh/WaveNet-VNNs-for-ANC
