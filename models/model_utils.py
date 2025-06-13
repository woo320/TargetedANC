"""
모델 관련 유틸리티 함수들 (추론 스타일 지원)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from config.constants import SR

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
    """SudoRM-RF 가중치 로드 유틸리티"""
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
    """WaveNet 가중치 로드 유틸리티"""
    state = torch.load(checkpoint_path, map_location='cpu')
    model_state = state['model'] if 'model' in state else state
    model.load_state_dict(model_state)
    return model

def load_broadcast_classifier_weights(model, checkpoint_path):
    """BroadcastClassifier 가중치 로드 유틸리티"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ BroadcastClassifier checkpoint not found: {checkpoint_path}")
        print("   Using randomly initialized weights")
        return model
    
    try:
        state = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"✅ BroadcastClassifier weights loaded from: {checkpoint_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading BroadcastClassifier weights: {e}")
        print("   Using randomly initialized weights")
        return model

def prepare_audio_for_classifier(audio, window_len=16000):
    """
    분류기용 오디오 전처리
    
    Args:
        audio: (B, C, T) 형태의 오디오
        window_len: 분류기 입력 길이
    
    Returns:
        processed_audio: (B, 1, window_len) 형태
    """
    # 차원 조정
    if audio.dim() == 3:
        audio = audio.squeeze(1)  # (B, T)
    elif audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, T)
    
    batch_size, seq_len = audio.shape
    
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

# ===== 🔄 분류 손실 함수들 =====

def compute_broadcast_classification_loss(classification_output, s1_separated, s2_separated, bce_loss_fn, device):
    """
    🔧 학습 코드 호환용 통합 분류 손실 함수
    추론 스타일과 텐서 스타일 모두 지원
    """
    if classification_output is None:
        return torch.tensor(0.0, device=device), 0.0
    
    # 📋 케이스 1: 추론 스타일 (딕셔너리)
    if isinstance(classification_output, dict) and 'batch_info' in classification_output:
        return compute_broadcast_classification_loss_inference_style(
            classification_output, s1_separated, s2_separated, bce_loss_fn, device
        )
    
    # 📋 케이스 2: 텐서 스타일 (기존 방식)
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
            print(f"⚠️ Tensor-style classification loss calculation failed: {e}")
            return torch.tensor(0.0, device=device), 0.0
    
    # 📋 케이스 3: 알 수 없는 형식
    else:
        print(f"⚠️ Unknown classification output type: {type(classification_output)}")
        return torch.tensor(0.0, device=device), 0.0

def compute_broadcast_classification_loss_inference_style(
    classification_results, s1_target, s2_target, bce_loss_fn, device
):
    """
    추론 방식의 분류 손실 계산
    
    Args:
        classification_results: joint model의 분류 결과
        s1_target: Ground truth s1 (방송)
        s2_target: Ground truth s2 (소음)
        bce_loss_fn: BCEWithLogitsLoss
        device: torch device
    
    Returns:
        classification_loss: 분류 손실
        accuracy: 정확도
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
        
        # 확률을 logit으로 변환 (안전한 방법)
        chan0_prob = max(0.001, min(0.999, chan0_prob))  # 클리핑
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
    분류 정확도에 따른 보상/페널티 계산
    
    Args:
        classification_results: joint model의 분류 결과
        device: torch device
    
    Returns:
        reward_penalty: 보상(음수)/페널티(양수) 값
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
            total_penalty -= 0.1  # 작은 보상
        else:
            total_penalty += 0.5   # 큰 페널티
    
    return torch.tensor(total_penalty / batch_size, device=device)

# ===== 기존 유틸리티 함수들 =====

def compute_anc_losses(s2_enhanced, s2_target, dba_loss_fn, nmse_loss_fn, device):
    """ANC 손실 계산 유틸리티"""
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
    """손실 가중치 계산"""
    if use_dynamic_mix:
        # SudoRM-RF: 분리 품질이 더 중요
        return {
            'final_quality': 0.25,     # 25%
            'anc_total': 0.35,         # 35%
            's1_separation': 0.3,      # 30% (증가)
            'antinoise_constraint': 0.1 # 10%
        }
    else:
        # 기존 pre-mixed: ANC 중심
        return {
            'final_quality': 0.3,      # 30%
            'anc_total': 0.4,          # 40%
            's1_separation': 0.2,      # 20%
            'antinoise_constraint': 0.1 # 10%
        }

def create_loss_functions(device, fs=16000, nfft=512):
    """손실 함수들 생성"""
    # WaveNet 경로 추가
    sys.path.insert(0, "/content/drive/MyDrive/joint/WaveNet-VNNs-for-ANC/WaveNet_VNNs")
    from loss_function import dBA_Loss, NMSE_Loss
    
    dba_loss = dBA_Loss(fs=fs, nfft=nfft, f_up=fs/2).to(device)
    nmse_loss = NMSE_Loss().to(device)
    
    return dba_loss, nmse_loss

def setup_optimizer_groups(separation_model, noise_model, eta_param, config):
    """옵티마이저 파라미터 그룹 설정"""
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
    """학습 단계 정의"""
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
    """오디오 길이 보정 유틸리티"""
    import numpy as np
    
    if isinstance(arr, np.ndarray) and arr.ndim > 0:
        return arr[:target_len] if len(arr) >= target_len else np.pad(arr, (0, target_len - len(arr)), 'constant')
    else:
        return np.full(target_len, float(arr) if np.isscalar(arr) else 0.0)

def normalize_audio_for_save(audio_array, target_db=-20):
    """저장용 오디오 정규화"""
    import numpy as np
    
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
    """오버피팅 체크 (음수 손실 고려)"""
    return abs(train_metric) > abs(val_metric) * threshold

def calculate_snr_improvement(input_audio, enhanced_audio, eps=1e-9):
    """SNR 개선도 계산"""
    input_power = torch.mean(input_audio ** 2)
    enhanced_power = torch.mean(enhanced_audio ** 2)
    snr_improvement = 10 * torch.log10(enhanced_power / (input_power + eps))
    return snr_improvement.item()

def format_memory_info(allocated_gb, reserved_gb, total_gb=None):
    """메모리 정보 포맷팅"""
    info = f"{allocated_gb:.1f}GB allocated, {reserved_gb:.1f}GB reserved"
    if total_gb:
        info += f" (Total: {total_gb:.1f}GB)"
    return info

def setup_cuda_environment():
    """CUDA 환경 최적화 설정"""
    import os
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 효율적인 메모리 할당 전략
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
        
        return True
    return False

def get_device_info():
    """디바이스 정보 반환"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return device_name, total_memory
    else:
        return "CPU", 0.0