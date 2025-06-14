"""
안내방송음 /그 외 소음 판별 이진 분류기 블록
"""
import torch
import torch.nn.functional as F
import sys
import os
import numpy as np


# WaveNet-VNNs 경로 추가
sys.path.insert(0, "/content/drive/MyDrive/joint/WaveNet-VNNs-for-ANC/WaveNet_VNNs")
from loss_function import dBA_Loss, NMSE_Loss

def sisdr_loss(est, target, zero_mean=True, eps=1e-9):
    """SI-SDR loss 계산"""
    if est.dim() == 3:
        est = est.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    if zero_mean:
        est = est - est.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

    dot = torch.sum(est * target, dim=-1, keepdim=True)
    energy = torch.sum(target**2, dim=-1, keepdim=True) + eps
    proj = (dot / energy) * target
    noise = est - proj

    sisdr = 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))
    return -sisdr.mean()

def load_sudormrf_weights(model, checkpoint_path):
    """C-SudoRM-RF++ 가중치 로드"""
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in state:
        actual_state = state['model_state_dict']
    elif 'model' in state:
        actual_state = state['model']
    else:
        actual_state = state

    # module. 접두사 제거
    if any(key.startswith('module.') for key in actual_state.keys()):
        new_state = {key[7:] if key.startswith('module.') else key: value
                    for key, value in actual_state.items()}
        actual_state = new_state

    model.load_state_dict(actual_state)
    return model

def load_wavenet_weights(model, checkpoint_path):
    """WaveNet-VNNs 가중치 로드"""
    state = torch.load(checkpoint_path, map_location='cpu')
    model_state = state['model'] if 'model' in state else state
    model.load_state_dict(model_state)
    return model

def load_broadcast_classifier_weights(model, checkpoint_path):
    """BroadcastClassifier 가중치 로드"""
    if not os.path.exists(checkpoint_path):
        print(f" BroadcastClassifier checkpoint not found: {checkpoint_path}")
        print("   Using randomly initialized weights")
        return model
    
    try:
        state = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state)
        print(f" BroadcastClassifier weights loaded from: {checkpoint_path}")
        return model
    except Exception as e:
        print(f" Error loading BroadcastClassifier weights: {e}")
        print("   Using randomly initialized weights")
        return model

def prepare_audio_for_classifier(audio, window_len=16000):
    """
    BroadcastClassifier(분류기) 입력 오디오 전처리
    입력 shape, 길이 표준화 ((B,1,T) or (B,T)→(B,1,window_len)), 중앙 부분 추출 or zero-padding
    분류기가 입력받는 고정 길이 텐서로 맞춰줌
    """
    # 차원 조정
    if audio.dim() == 3:
        audio = audio.squeeze(1)  # (B, T)
    elif audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, T)
    
    seq_len = audio.shape
    
    # 길이 조정
    if seq_len > window_len:
        # 중앙 부분 추출
        start_idx = (seq_len - window_len) // 2
        audio = audio[:, start_idx:start_idx + window_len]
    elif seq_len < window_len:
        # 패딩
        pad_len = window_len - seq_len
        audio = F.pad(audio, (0, pad_len), mode='constant', value=0)
    
    # (B, 1, T) 형태로 변환
    return audio.unsqueeze(1)



# =====  분류 손실 함수들 =====
def compute_broadcast_classification_loss(classification_output, s1_separated, s2_separated, bce_loss_fn, device):
    """
    BroadcastClassifier의 이진 분류 손실 및 정확도 계산
    분리된 음성/소음의 에너지 비교로 라벨 생성
    BCE(Binary Cross Entropy) 손실과 예측 확률(>0.5)로 정확도 산출
    입력: 분류기 출력(딕셔너리 or 텐서), 정답 s1/s2, 손실함수, 디바이스
    출력: 손실값, 정확도
    """
    if classification_output is None:
        return torch.tensor(0.0, device=device), 0.0
    
    #케이스 1: 추론 스타일 (딕셔너리)
    if isinstance(classification_output, dict) and 'batch_info' in classification_output:
        return compute_broadcast_classification_loss_inference_style(
            classification_output, s1_separated, s2_separated, bce_loss_fn, device
        )
    
    #케이스 2: 텐서 스타일 (기존 방식)
    elif hasattr(classification_output, 'shape') and hasattr(classification_output, 'dim'):
        try:
            # 타겟 생성: s1과 s2의 에너지 비교
            s1_energy = torch.mean(s1_separated.abs(), dim=-1)
            s2_energy = torch.mean(s2_separated.abs(), dim=-1)
            targets = (s1_energy > s2_energy).float().squeeze(-1)
            
            # BCE 손실 계산
            classification_loss = bce_loss_fn(
                classification_output.squeeze(-1) if classification_output.dim() > 1 else classification_output,
                targets
            )
            
            # 정확도 계산
            probs = torch.sigmoid(classification_output.squeeze())
            predicted = (probs > 0.5).float()
            accuracy = (predicted == targets).float().mean().item()
            
            return classification_loss, accuracy
            
        except Exception as e:
            print(f" Tensor-style classification loss calculation failed: {e}")
            return torch.tensor(0.0, device=device), 0.0
    
    else:
        print(f" Unknown classification output type: {type(classification_output)}")
        return torch.tensor(0.0, device=device), 0.0

def compute_broadcast_classification_loss_inference_style(
    classification_results, s1_target, s2_target, bce_loss_fn, device
):
    """
    추론(딕셔너리 형태) 분류 결과에 대한 BCE 손실, 정확도 계산
    확률→logit 변환, 각 채널별로 정답 라벨 지정(s1: 1, s2: 0)
    평균 손실과 batch accuracy 반환
    """
    if 'batch_info' not in classification_results:
        return torch.tensor(0.0, device=device), 0.0
    
    batch_info = classification_results['batch_info']
    batch_size = len(batch_info)
    
    total_loss = 0.0
    correct_classifications = 0
    
    for b, info in enumerate(batch_info):
        # Ground truth: s1은 방송(1), s2는 소음(0)
        chan0_prob = info['mask_chan0']  # 채널0의 방송 확률
        chan1_prob = info['mask_chan1']  # 채널1의 방송 확률
        
        # 확률을 logit으로 변환
        chan0_prob = max(0.001, min(0.999, chan0_prob))
        chan1_prob = max(0.001, min(0.999, chan1_prob))
        
        chan0_logit = torch.log(torch.tensor(chan0_prob / (1 - chan0_prob), device=device))
        chan1_logit = torch.log(torch.tensor(chan1_prob / (1 - chan1_prob), device=device))
        
        # Ground truth 라벨
        chan0_target = torch.tensor(1.0, device=device)  # s1은 방송
        chan1_target = torch.tensor(0.0, device=device)  # s2는 소음
        
        # BCE 손실 계산
        loss0 = bce_loss_fn(chan0_logit.unsqueeze(0), chan0_target.unsqueeze(0))
        loss1 = bce_loss_fn(chan1_logit.unsqueeze(0), chan1_target.unsqueeze(0))
        
        total_loss += (loss0 + loss1) / 2
        
        # 정확도: 채널0(s1)이 더 높은 확률을 가져야 함
        predicted_correctly = (chan0_prob > chan1_prob)
        if predicted_correctly:
            correct_classifications += 1
    
    avg_loss = total_loss / batch_size
    accuracy = correct_classifications / batch_size
    
    return avg_loss, accuracy

def compute_classification_reward_penalty(classification_results, device):
    """
    분류 결과에 따른 보상/페널티 산출(보조 손실용)
    정답 예측 시 보상(-0.1), 오답시 페널티(+0.5)
    """
    if not isinstance(classification_results, dict) or 'batch_info' not in classification_results:
        return torch.tensor(0.0, device=device)
    
    batch_info = classification_results['batch_info']
    batch_size = len(batch_info)
    
    total_penalty = 0.0
    
    for info in batch_info:
        chan0_prob = info['mask_chan0']
        chan1_prob = info['mask_chan1']
        
        # Ground truth: 채널0(s1)이 방송, 채널1(s2)이 소음
        correct_classification = (chan0_prob > chan1_prob)
        
        if correct_classification:
            total_penalty -= 0.1
        else:
            total_penalty += 0.5
    
    return torch.tensor(total_penalty / batch_size, device=device)

# 기존 유틸리티 함수들

def compute_anc_losses(s2_enhanced, s2_target, dba_loss_fn, nmse_loss_fn, device):
    """
    ANC 결과 신호에 대한 loss(dBA, NMSE) 계산
    dBA: 소리의 청감 특성 반영 손실, NMSE: 정규화된 평균 제곱 오차
    WaveNet-VNNs 저감 결과의 정량적 성능 평가/학습에 활용
    """
    try:
        s2_enhanced_flat = s2_enhanced.squeeze(1)
        s2_target_flat = s2_target.squeeze(1)

        anc_dba_loss = dba_loss_fn(s2_enhanced_flat) - dba_loss_fn(s2_target_flat)
        anc_nmse_loss = nmse_loss_fn(s2_enhanced_flat, s2_target_flat)
    except Exception:
        anc_dba_loss = torch.tensor(0.0, device=device)
        anc_nmse_loss = torch.tensor(0.0, device=device)
    
    return anc_dba_loss, anc_nmse_loss

def compute_loss_weights(use_dynamic_mix=False):
    """
    파이프라인 전체 손실 가중치 설정
    분리/저감/합성 등 여러 loss의 가중치를 상황(실험목적)에 맞게 배분
    """
    if use_dynamic_mix:
        # C-SudoRM-RF++: 분리 중심
        return {
            'final_quality': 0.25,   
            'anc_total': 0.35,       
            's1_separation': 0.3,
            'antinoise_constraint': 0.1
        }
    else:
        # 기존 pre-mixed: ANC 중심
        return {
            'final_quality': 0.3,     
            'anc_total': 0.4,          
            's1_separation': 0.2,     
            'antinoise_constraint': 0.1
        }

def create_loss_functions(device, fs=16000, nfft=512):
    """
    dBA, NMSE 등 WaveNet-VNNs 논문 기반 손실 함수 객체 생성
    """
    
    
    dba_loss = dBA_Loss(fs=fs, nfft=nfft, f_up=fs/2).to(device)
    nmse_loss = NMSE_Loss().to(device)
    
    return dba_loss, nmse_loss

def setup_optimizer_groups(separation_model, noise_model, eta_param, config):
    """분리/저감/eta 파라미터별 별도 학습률 등 옵티마이저 그룹 구성"""
    separation_params = list(separation_model.parameters())
    noise_params = list(noise_model.parameters())
    eta_params = [eta_param] if eta_param is not None else []

    param_groups = [
        {'params': separation_params, 'lr': config.get('separation_lr', 5e-6)},
        {'params': noise_params, 'lr': config.get('noise_lr', 1e-5)},
    ]

    if eta_params:
        param_groups.append({'params': eta_params, 'lr': config.get('eta_lr', 2e-5)})

    return param_groups

def get_training_stages(total_epochs):
    """
    학습 전체를 세 단계로 구분 (초기 적응, 안정화, 미세수렴)
    각 단계별 에폭 구간, 설명 문자열 포함
    """
    return {
        "Initial Adaptation": {
            "range": f"1-{min(3, total_epochs//3)}",
            "description": "Models learning to cooperate"
        },
        "Stabilization": {
            "range": f"{min(4, total_epochs//3 + 1)}-{min(total_epochs*2//3, total_epochs-2)}",
            "description": "Gradual ANC optimization"
        },
        "Fine Convergence": {
            "range": f"{max(total_epochs*2//3 + 1, total_epochs-1)}-{total_epochs}",
            "description": "Final tuning and convergence"
        }
    }

def get_current_stage(epoch, total_epochs):
    """현재 학습 단계 반환"""
    if epoch <= total_epochs // 3:
        return "Initial Adaptation"
    elif epoch <= total_epochs * 2 // 3:
        return "Stabilization"
    else:
        return "Fine Convergence"

def ensure_audio_length(arr, target_len):
    """
    오디오 배열 길이 보정
    넘파이 배열을 타겟 길이에 맞게 자르거나 0으로 패딩
    """
    
    if isinstance(arr, np.ndarray) and arr.ndim > 0:
        return arr[:target_len] if len(arr) >= target_len else np.pad(arr, (0, target_len - len(arr)), 'constant')
    else:
        return np.full(target_len, float(arr) if np.isscalar(arr) else 0.0)

def normalize_audio_for_save(audio_array, target_db=-20):
    """
    오디오 저장용 정규화
    wave파일 등으로 저장하기 전에 오디오 볼륨 맞추기
    """
    
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.squeeze().cpu().numpy()

    if len(audio_array) > 0:
        audio_power = np.mean(audio_array ** 2) + 1e-9
        audio_db = 10 * np.log10(audio_power)
        gain_db = target_db - audio_db
        gain_linear = 10 ** (gain_db / 20)
        normalized = np.clip(audio_array * gain_linear, -1.0, 1.0)
        return normalized
    else:
        return audio_array

def check_overfitting(train_metric, val_metric, threshold=1.3):
    """오버피팅 체크 (음수 손실 기준) 학습/검증 성능 차이 임계값 초과 시 True 반환"""
    return abs(train_metric) > abs(val_metric) * threshold

def calculate_snr_improvement(input_audio, enhanced_audio, eps=1e-9):
    """SNR(신호 대 잡음비) 개선도 계산"""
    input_power = torch.mean(input_audio ** 2)
    enhanced_power = torch.mean(enhanced_audio ** 2)
    snr_improvement = 10 * torch.log10(enhanced_power / (input_power + eps))
    return snr_improvement.item()

def format_memory_info(allocated_gb, reserved_gb, total_gb=None):
    """메모리 사용량 등 정보 포맷팅 문자열 반환"""
    info = f"{allocated_gb:.1f}GB allocated, {reserved_gb:.1f}GB reserved"
    if total_gb:
        info += f" (Total: {total_gb:.1f}GB)"
    return info

def setup_cuda_environment():
    """CUDA 성능 최적화 및 결정론적 환경 설정"""
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
        
        return True
    return False

def get_device_info():
    """현재 사용중인 디바이스(GPU/CPU) 이름 및 전체 메모리 반환"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return device_name, total_memory
    else:
        return "CPU", 0.0