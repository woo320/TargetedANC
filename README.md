# TargetedANC



## Model Architecture
![image](https://github.com/user-attachments/assets/4b1b942b-d494-4a87-baa0-9376238e0d46)

## Setup
```
pip install -r requirements.txt
```

실행 환경: Pyhton: 3.11.13 / Torch: 2.6.0 / Torchaudio: 2.6.0 / librosa: 0.11.0 / Soundfile: 0.13.1

---




# Inference_Pipeline
## **분리, 분류, 저감 모델의 Training과 Inference Command line usage**

### C-SuDoRM-RF++ Training Command
```bash
!python /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/c_sudormrf_train.py \
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
| 옵션명                | 설명                                                      |
|----------------------|-----------------------------------------------------------|
| --model_type         | 사용 모델 타입 (causal이면 C-SuDORMRF++ 사용)             |
| --train              | 학습 데이터셋 이름                                        |
| --val                | 검증 데이터셋 이름                                        |
| --test               | 테스트 데이터셋 이름                                      |
| --n_channels         | 입력 오디오 채널 수                                       |
| --fs                 | 샘플링레이트(Hz)                                          |
| --batch_size         | 배치 크기                                                 |
| --n_epochs           | 학습 에폭 수                                              |
| --audio_timelength   | 한 샘플 길이(초)                                          |
| --enc_kernel_size    | 인코더 커널 크기                                          |
| --enc_num_basis      | 인코더 basis 수                                           |
| --in_channels        | 내부 채널 수                                              |
| --out_channels       | 외부 채널 수                                              |
| --num_blocks         | 블록 수                                                   |
| --lr                 | 학습률(learning rate)                                     |
| --divide_lr_by       | 학습률 감소 비율                                          |
| --patience           | validation patience                                       |
| --early_stop_patience| early stopping patience                                   |
| --upsampling_depth   | 업샘플링 depth                                            |
| --max_num_sources    | 최대 소스 수                                              |
| --min_num_sources    | 최소 소스 수                                              |
| --zero_pad_audio     | 오디오 zero pad 사용 여부(플래그)                         |
| --normalize_audio    | 오디오 normalize 사용 여부(플래그)                        |
| --cad                | 사용 GPU id                                               |
| --n_jobs             | 병렬처리 쓰레드 수                                        |
| --clp                | 체크포인트 저장 폴더 경로                                 |



### C-SuDoRM-RF++ Inference Command
```bash
!python /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/c_sudormrf_inference.py \
  -ckpt /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/causal_best.pt \
  --input_dir <your_data_dir> \
  --output_dir <your_output_dir>
```

---

### Audio Segment Classifier(ASC) Training Command
```bash
!python /content/drive/MyDrive/inference_pipeline/ASC/ASC.py \
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

---

### WaveNet-VNNs Training Command
```bash
!python /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/train_opt_210.py \
  --config /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/cfg_train_opt_210.toml \
  --device 0
```

### WaveNet-VNNs Infernce Command
```bash
!python /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/inference_opt.py \
  --model-path /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/model.pth \
  --config /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/config_opt_210.json \
  --test-data-dir <your_data_dir> \
  --output-enh-dir <your_denoise_dir> \
  --output-anti-dir <your_antinoise_dir>
```

---

  ## EndtoEnd Inference Command
```bash
!python /content/drive/MyDrive/inference_pipeline/end2end_inference.py \
  --sep_ckpt      /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/causal_best.pt \
  --noise_cfg     /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/config_opt_210.json \
  --noise_ckpt   /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/model.pth \
  --bcd_ckpt      /content/drive/MyDrive/inference_pipeline/ASC/asc.pth \
  --input_dir     <your_data_dir> \
  --sep_out      <your_seperation_dir> \
  --noise_out       <your_noise_dir> \
  --denoise_out      <your_denoise_dir> \
  --anti_out     <your_antinoise_dir> \
  --final_out   <your_final_dir>
```

**모든 out 및 output 인자들은 빈 디렉토리로 미리 준비해두어야합니다.**
