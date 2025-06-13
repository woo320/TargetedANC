#!/usr/bin/env python3
import os
import sys
import time
import glob
import json

import torch
import numpy as np
import soundfile as sf

# ─── 경로 설정 ─────────────────────────────────────────────────────────
# 조인트 모델 경로 추가
sys.path.append('/content/drive/MyDrive/joint/code4')
# WaveNet-VNNs 모듈 경로
sys.path.append('/content/drive/MyDrive/WaveNet-VNNs-for-ANC/WaveNet_VNNs')
# sudo_rm_rf 분리 모델 경로
sys.path.append('/content/drive/MyDrive/sudo_rm_rf')

# ─── 조인트 모델 import ─────────────────────────────────────────────
from models.joint_model import ImprovedJointModel
from project_utils.audio_utils import standardize_audio_dims
from config.constants import SR

# ─── 기존 필요한 모듈들 ───
from loss_function import dBA_Loss


def load_joint_model(joint_checkpoint_path, device):
    """조인트 모델 체크포인트에서 모델 로드"""
    print(f"🔧 Loading joint model from: {joint_checkpoint_path}")

    # 체크포인트 로드
    checkpoint = torch.load(joint_checkpoint_path, map_location=device, weights_only=False)

    # 설정 정보 추출
    if 'config' in checkpoint:
        config = checkpoint['config']
        sudormrf_checkpoint = config['sudormrf_checkpoint']
        wavenet_checkpoint = config['wavenet_checkpoint']
        wavenet_config = config['wavenet_config']
        broadcast_classifier_checkpoint = config.get('broadcast_classifier_checkpoint')
        use_broadcast_classifier = config.get('use_broadcast_classifier', True)
    else:
        # 기본 경로들 (체크포인트에 config가 없는 경우)
        print("⚠️ No config in checkpoint, using default paths")
        sudormrf_checkpoint = "/content/drive/MyDrive/joint/weight/.pt"
        wavenet_checkpoint = "/content/drive/MyDrive/joint/weight/reduction.pth"
        wavenet_config = "/content/drive/MyDrive/joint/WaveNet-VNNs-for-ANC/WaveNet_VNNs/config_opt_210.json"
        broadcast_classifier_checkpoint = "/content/drive/MyDrive/joint/weight/classifier.pth"
        use_broadcast_classifier = True

    # 조인트 모델 생성
    model = ImprovedJointModel(
        sudormrf_checkpoint_path=sudormrf_checkpoint,
        wavenet_checkpoint_path=wavenet_checkpoint,
        wavenet_config_path=wavenet_config,
        broadcast_classifier_checkpoint_path=broadcast_classifier_checkpoint,
        use_broadcast_classifier=use_broadcast_classifier
    ).to(device)

    # 조인트 모델 가중치 로드
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        model_state = checkpoint

    # module. 접두사 제거 (DataParallel 사용했던 경우)
    if any(key.startswith('module.') for key in model_state.keys()):
        new_state = {key[7:] if key.startswith('module.') else key: value
                    for key, value in model_state.items()}
        model_state = new_state

    model.load_state_dict(model_state, strict=False)  # strict=False로 일부 누락 허용
    model.eval()

    print(f"✅ Joint model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if 'anc_total' in metrics:
            print(f"   ANC Loss: {metrics['anc_total']:.4f}")
        if 'classification_accuracy' in metrics:
            print(f"   Classification Accuracy: {metrics['classification_accuracy']:.3f}")

    return model


def run_joint_inference(
    joint_ckpt,
    input_dir,
    sep_out,
    noise_out,
    anti_out,
    denoise_out,
    final_out,
    use_inference_style=True,
    save_debug=False,
    skip_first_n=3
):
    """조인트 모델을 사용한 end-to-end 추론 (WaveNet 스타일 NMSE)"""

    # 출력 디렉토리 생성
    os.makedirs(sep_out,      exist_ok=True)
    os.makedirs(noise_out,    exist_ok=True)
    os.makedirs(anti_out,     exist_ok=True)
    os.makedirs(denoise_out,  exist_ok=True)
    os.makedirs(final_out,    exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # 조인트 모델 로드
    joint_model = load_joint_model(joint_ckpt, device)

    # dBA 손실 함수 (지표 계산용)
    try:
        dBA_fn = dBA_Loss(fs=SR, nfft=512, f_up=SR/2).to(device)
    except Exception as e:
        print(f"⚠️ dBA Loss function failed: {e}, using simple metric")
        dBA_fn = None

    # 청크 크기 설정 (메모리에 따라 조정)
    chunk_len = int(4.0 * SR)  # 4초

    print(f"🎯 Processing audio files...")
    print(f"   Input directory: {input_dir}")
    print(f"   Chunk length: {chunk_len/SR:.1f} seconds")
    print(f"   BroadcastClassifier: {'Enabled' if joint_model.use_broadcast_classifier else 'Disabled'}")
    print(f"   Inference style: {'Inference' if use_inference_style else 'Training'}")

    # 입력 파일 찾기
    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if not wav_files:
        print(f"❌ No .wav files found in {input_dir}")
        return

    metrics = {
        'dBA': [],
        'nmse': [],           # WaveNet 스타일 NMSE
        'si_sdr': [],
        'snr_improvement': [], # WaveNet 스타일 SNR
        'rtf': [],
        'ms_per_s': []        # 🆕 처리 속도 (ms/s) 추가
    }

    for idx, path in enumerate(wav_files):
        fname = os.path.splitext(os.path.basename(path))[0]
        print(f"\n🔄 Processing: {fname}")

        try:
            # 오디오 로드
            wav, fs = sf.read(path)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)  # 모노로 변환

            if fs != SR:
                print(f"   ⚠️ Sample rate mismatch: expected {SR}, got {fs}")
                # 리샘플링 시도 (librosa 사용)
                try:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=fs, target_sr=SR)
                    print(f"   ✅ Resampled to {SR}Hz")
                except ImportError:
                    print(f"   ❌ Cannot resample (librosa not available)")
                    continue

            t0 = time.time()

            # 입력 오디오 준비
            audio_tensor = torch.from_numpy(wav).float().to(device)
            audio_tensor = standardize_audio_dims(audio_tensor)  # [1, 1, T] 형태로 변환

            print(f"   Input shape: {audio_tensor.shape}")
            print(f"   Audio duration: {len(wav)/SR:.2f} seconds")

            # 🚀 조인트 모델 추론 (한 번에 모든 처리 완료!)
            with torch.no_grad():
                if use_inference_style:
                    # 추론 스타일 (더 안정적)
                    outputs = joint_model.forward_inference_style(
                        audio_tensor,
                        chunk_len=chunk_len,
                        return_classification=True
                    )
                else:
                    # 학습 스타일 (더 빠름)
                    outputs = joint_model.forward_for_training(
                        audio_tensor,
                        chunk_len=chunk_len,
                        return_classification=True
                    )

            t1 = time.time()

            # 결과 추출
            sep_broadcast = outputs['s1_clean'].squeeze().cpu().numpy()      # 분리된 방송
            sep_noise = outputs['s2_noise'].squeeze().cpu().numpy()          # 분리된 소음 (원본)
            s2_target = outputs['s2_target'].squeeze().cpu().numpy()         # ANC 타겟
            s2_antinoise = outputs['s2_antinoise'].squeeze().cpu().numpy()   # ANC 상쇄 신호
            s2_enhanced = outputs['s2_enhanced'].squeeze().cpu().numpy()     # ANC 처리된 소음
            final_mix = outputs['enhanced_verification'].squeeze().cpu().numpy()  # 최종 결과

            # 분류 결과 처리
            classification_info = ""
            if 'classification' in outputs and outputs['classification'] is not None:
                try:
                    classification_results = outputs['classification']
                    
                    if isinstance(classification_results, dict) and 'batch_info' in classification_results:
                        # 추론 스타일 결과
                        batch_info = classification_results['batch_info'][0]
                        
                        # ✅ 올바른 키 사용
                        chan0_prob = batch_info['ch0_prob']  # mask_chan0 → ch0_prob
                        chan1_prob = batch_info['ch1_prob']  # mask_chan1 → ch1_prob
                        is_ch0_broadcast = batch_info['is_channel0_broadcast']
                        
                        classification_info = f"Ch0:{chan0_prob:.3f}, Ch1:{chan1_prob:.3f}, Broadcast:Ch{'0' if is_ch0_broadcast else '1'}"
                        
                    elif isinstance(classification_results, torch.Tensor):
                        prob = torch.sigmoid(classification_results).item()
                        classification_info = f"Broadcast prob: {prob:.3f}"
                    else:
                        classification_info = "Classification available"
                        
                except KeyError as e:
                    print(f"⚠️ KeyError: {e}")
                    classification_info = "Classification key error"
                except Exception as e:
                    print(f"⚠️ Classification error: {e}")
                    classification_info = "Classification failed"

            '''classification_info = ""
                if 'classification' in outputs and outputs['classification'] is not None:
                if isinstance(outputs['classification'], dict) and 'batch_info' in outputs['classification']:
                    # 추론 스타일 결과
                    batch_info = outputs['classification']['batch_info'][0]  # 첫 번째 배치
                    chan0_prob = batch_info['mask_chan0']
                    chan1_prob = batch_info['mask_chan1']
                    is_ch0_broadcast = batch_info['is_channel0_broadcast']
                    classification_info = f"Ch0:{chan0_prob:.3f}, Ch1:{chan1_prob:.3f}, Broadcast:Ch{'0' if is_ch0_broadcast else '1'}"
                elif isinstance(outputs['classification'], torch.Tensor):
                    # 학습 스타일 결과
                    prob = torch.sigmoid(outputs['classification']).item()
                    classification_info = f"Broadcast prob: {prob:.3f}"'''

            # 성능 지표 계산
            total_ms = (t1 - t0) * 1000.0
            audio_sec = len(wav) / SR
            rtf = (total_ms / 1000.0) / audio_sec
            ms_per_s = total_ms / audio_sec  # 이미 계산되어 있음

            # 🔄 WaveNet 스타일 지표 계산
            try:
                if dBA_fn is not None:
                    enh_t = torch.tensor(s2_enhanced, device=device).unsqueeze(0)
                    tgt_t = torch.tensor(s2_target, device=device).unsqueeze(0)

                    dBA_val = (dBA_fn(enh_t) - dBA_fn(tgt_t)).item()

                    # 🎯 WaveNet 스타일 NMSE 계산
                    nmse_val = 10 * torch.log10(
                        torch.sum(enh_t.squeeze() ** 2) /
                        torch.sum(tgt_t.squeeze() ** 2)
                    ).item()

                    # SI-SDR 계산 (기존 유지)
                    def compute_si_sdr(est, ref):
                        est_zm = est - torch.mean(est)
                        ref_zm = ref - torch.mean(ref)
                        alpha = torch.sum(est_zm * ref_zm) / torch.sum(ref_zm ** 2)
                        ref_scaled = alpha * ref_zm
                        noise = est_zm - ref_scaled
                        sisdr = 10 * torch.log10(torch.sum(ref_scaled ** 2) / (torch.sum(noise ** 2) + 1e-9))
                        return sisdr.item()

                    si_sdr_val = compute_si_sdr(enh_t.squeeze(), tgt_t.squeeze())

                    # 🎯 WaveNet 스타일 SNR 계산 (출력 SNR)
                    snr_improvement = 10 * torch.log10(
                        torch.sum(tgt_t.squeeze() ** 2) /
                        (torch.sum((enh_t.squeeze() - tgt_t.squeeze()) ** 2) + 1e-9)
                    ).item()

                else:
                    # 간단한 numpy 계산
                    nmse_val = 10 * np.log10(np.sum(s2_enhanced ** 2) / np.sum(s2_target ** 2))
                    dBA_val = 20 * np.log10(np.sqrt(np.mean((s2_enhanced - s2_target) ** 2)) + 1e-9)
                    si_sdr_val = 0.0
                    snr_improvement = 0.0

            except Exception as e:
                print(f"   ⚠️ Metric calculation failed: {e}")
                dBA_val = 0.0
                nmse_val = 0.0
                si_sdr_val = 0.0
                snr_improvement = 0.0

            # ─────────── 파일 저장 ───────────
            sf.write(os.path.join(sep_out,     f"{fname}_broadcast.wav"), sep_broadcast, SR)
            sf.write(os.path.join(noise_out,   f"{fname}_sep_noise.wav"), sep_noise, SR)
            sf.write(os.path.join(anti_out,    f"{fname}_anti_noise.wav"), s2_antinoise, SR)
            sf.write(os.path.join(denoise_out, f"{fname}_denoised_noise.wav"), s2_enhanced, SR)
            sf.write(os.path.join(final_out,   f"{fname}_final.wav"), final_mix, SR)

            # ✅ metrics 저장 (중복 제거 - 한 번만!)
            if idx >= skip_first_n:
                metrics['dBA'].append(dBA_val)
                metrics['nmse'].append(nmse_val)        # WaveNet 스타일
                metrics['si_sdr'].append(si_sdr_val)
                metrics['snr_improvement'].append(snr_improvement)  # WaveNet 스타일
                metrics['rtf'].append(rtf)
                metrics['ms_per_s'].append(ms_per_s)    # 🆕 처리 속도 추가

            # 추가 저장 (디버깅용)
            if save_debug:
                debug_dir = os.path.join(final_out, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                sf.write(os.path.join(debug_dir, f"{fname}_s2_target.wav"), s2_target, SR)
                sf.write(os.path.join(debug_dir, f"{fname}_input.wav"), wav, SR)

            print(f"✅ {fname}: {total_ms:.1f}ms | RTF {rtf:.3f} | {ms_per_s:.1f}ms/s")
            print(f"   dBA: {dBA_val:+.2f}dB | NMSE: {nmse_val:+.2f}dB | SI-SDR: {si_sdr_val:+.2f}dB")
            print(f"   Output SNR: {snr_improvement:+.2f}dB")
            if classification_info:
                print(f"   Classification: {classification_info}")

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🎉 Joint model inference completed!")

    if len(metrics['dBA']) > 0:
        print(f"\n📈 PERFORMANCE METRICS (Average of {len(metrics['dBA'])} files)")
        print(f"   (Skipped first {skip_first_n} files)")
        print("=" * 60)  # 길이 조금 늘림

        avg_dBA = sum(metrics['dBA']) / len(metrics['dBA'])
        avg_nmse = sum(metrics['nmse']) / len(metrics['nmse'])
        avg_si_sdr = sum(metrics['si_sdr']) / len(metrics['si_sdr'])
        avg_snr_improvement = sum(metrics['snr_improvement']) / len(metrics['snr_improvement'])
        avg_rtf = sum(metrics['rtf']) / len(metrics['rtf'])
        avg_ms_per_s = sum(metrics['ms_per_s']) / len(metrics['ms_per_s'])  # 🆕 처리 속도 평균

        # 🎯 Audio Quality Metrics
        print(f"🎯 Audio Quality Metrics:")
        print(f"   Average dBA:             {avg_dBA:+7.3f} dB")
        print(f"   Average NMSE (WaveNet):  {avg_nmse:+7.3f} dB")
        print(f"   Average SI-SDR:          {avg_si_sdr:+7.3f} dB")
        print(f"   Average Output SNR:      {avg_snr_improvement:+7.3f} dB")

        # ⚡ Performance Metrics
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average RTF:             {avg_rtf:7.4f}")
        print(f"   Average Processing Speed: {avg_ms_per_s:6.1f} ms/s")  # 🆕 추가

        # 📊 Range Statistics (처리 속도 범위도 추가)
        print(f"\n📊 Range Statistics:")
        print(f"   dBA Range:      {min(metrics['dBA']):+.3f} ~ {max(metrics['dBA']):+.3f} dB")
        print(f"   NMSE Range:     {min(metrics['nmse']):+.3f} ~ {max(metrics['nmse']):+.3f} dB")
        print(f"   SI-SDR Range:   {min(metrics['si_sdr']):+.3f} ~ {max(metrics['si_sdr']):+.3f} dB")
        print(f"   RTF Range:      {min(metrics['rtf']):.4f} ~ {max(metrics['rtf']):.4f}")
        print(f"   Speed Range:    {min(metrics['ms_per_s']):.1f} ~ {max(metrics['ms_per_s']):.1f} ms/s")  # 🆕 추가

        print("=" * 60)

        # 🆕 JSON 형태도 업데이트
        summary = {
            'files_processed': len(wav_files),
            'files_for_average': len(metrics['dBA']),
            'skipped_files': skip_first_n,
            'audio_quality': {
                'dBA_dB': round(avg_dBA, 3),
                'nmse_dB': round(avg_nmse, 3),
                'si_sdr_dB': round(avg_si_sdr, 3),
                'output_snr_dB': round(avg_snr_improvement, 3)
            },
            'performance': {
                'rtf': round(avg_rtf, 4),
                'processing_speed_ms_per_s': round(avg_ms_per_s, 1)  # 🆕 추가
            }
        }

        print(f"\n📋 Summary (JSON format):")
        import json
        print(json.dumps(summary, indent=2))

    print(f"   Processed {len(wav_files)} files")
    print(f"   Output directories:")
    print(f"     - Broadcast: {sep_out}")
    print(f"     - Noise: {noise_out}")
    print(f"     - Anti-noise: {anti_out}")
    print(f"     - Denoised: {denoise_out}")
    print(f"     - Final: {final_out}")


# ⚡ Colab용 간편 함수들
def quick_inference(joint_model_path, input_audio_dir, output_base_dir, skip_first=3):
    """가장 간단한 추론 실행"""

    # 출력 디렉토리 자동 생성
    output_dirs = {
        'sep_out': os.path.join(output_base_dir, "broadcast"),
        'noise_out': os.path.join(output_base_dir, "noise"),
        'anti_out': os.path.join(output_base_dir, "antinoise"),
        'denoise_out': os.path.join(output_base_dir, "denoised"),
        'final_out': os.path.join(output_base_dir, "final")
    }

    print("🚀 Quick Joint Model Inference")
    print("=" * 50)
    print(f"Model: {joint_model_path}")
    print(f"Input: {input_audio_dir}")
    print(f"Output: {output_base_dir}")
    print("=" * 50)

    run_joint_inference(
        joint_ckpt=joint_model_path,
        input_dir=input_audio_dir,
        use_inference_style=True,
        save_debug=True,
        skip_first_n=skip_first,  #스
        **output_dirs
    )


# 🎯 사용 예시
if __name__ == "__main__":
    # 경로 설정 (실제 사용시 수정 필요)
    JOINT_MODEL_PATH = "/content/drive/MyDrive/joint/result/joint_mixed_training_2025-06-09-12h00m/weights/best_composite.pth"
    INPUT_DIR = "/content/drive/MyDrive/final_data/지하철실제녹음"
    OUTPUT_DIR = "/content/drive/MyDrive/joint/inference_output"

    # 실행
    quick_inference(JOINT_MODEL_PATH, INPUT_DIR, OUTPUT_DIR)