# TargetedANC



# 모델 아키텍처
![image](https://github.com/user-attachments/assets/4b1b942b-d494-4a87-baa0-9376238e0d46)

# 설치
```
pip install -r requirements.txt
```

실행 환경: Pyhton: 3.11.13 / Torch: 2.6.0 / Torchaudio: 2.6.0 / librosa: 0.11.0 / Soundfile: 0.13.1

# 실행




# Inference_Pipeline
## 분리, 분류, 저감 모델의 Training과 Inference Command line usage

### C-SuDoRM-RF++ Train Code
```bash
!python /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/c_sudormrf_train.py --model_type causal --train "ANNOUNCENOISE" --val "ANNOUNCENOISE" --test "ANNOUNCENOISE" --n_channels 1 -fs 16000 --batch_size 8 --n_epochs 200 --audio_timelength 4. --enc_kernel_size 21 --enc_num_basis 256 --in_channels 512 --out_channels 256 --num_blocks 18 -lr 0.001 --divide_lr_by 3. --patience 10 --early_stop_patience 30 --upsampling_depth 5 --max_num_sources 2 --min_num_sources 2 --zero_pad_audio --normalize_audio -cad 0 --n_jobs 4 -clp 'your_root'
```
### C-SuDoRM-RF++ Inference Code
```bash
!!python /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/c_sudormrf_inference.py -ckpt /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/causal_best.pt --input_dir 'your_data' --output_dir 'your_root'
```

### Audio Segment Classifier(ASC) Train Code
```bash
!python /content/drive/MyDrive/inference_pipeline/ASC/ASC.py \
    --train_s1_dir /content/drive/MyDrive/final_data/train/spk1 \
    --train_s2_dir /content/drive/MyDrive/final_data/train/spk2 \
    --val_s1_dir     /content/drive/MyDrive/final_data/val/spk1 \
    --val_s2_dir     /content/drive/MyDrive/final_data/val/spk2 \
    --test_s1_dir    /content/drive/MyDrive/final_data/test/spk1 \
    --test_s2_dir    /content/drive/MyDrive/final_data/test/spk2 \
    --save_path /content/drive/MyDrive/inference_pipeline/ASC/checkpoint/best.pth \
    --sr 16000 \
    --window_len 16000 \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 15
```
