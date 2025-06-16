import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
import numpy as np
from scipy.io import loadmat

# 프로젝트 경로 설정
sys.path.insert(0, "./Benchmark_models/c_sudorm_rf")
sys.path.insert(0, "./Benchmark_models/WaveNet_VNNs")

# C-SudoRM-RF++ 관련 import
from c_sudorm_rf.causal_improved_sudormrf_v3 import CausalSuDORMRF
import c_sudorm_rf.mixture_consistency as mixture_consistency

# WaveNet-VNNs 관련 anc 관련 import
from networks import WaveNet_VNNs
from utils import fir_filter, SEF

# 로컬 모듈 import
from project_utils.audio_utils import standardize_audio_dims
from config.constants import SR

# BroadcastClassifier 분류기 import
try:
    from models.broadcast_classifier import BroadcastClassifier
    from models.model_utils import load_broadcast_classifier_weights
except ImportError:
    try:
        from .broadcast_classifier import BroadcastClassifier
        from .model_utils import load_broadcast_classifier_weights
    except ImportError:
        BroadcastClassifier = None


"""Joint 모델 클래스 정의"""

class ImprovedJointModel(nn.Module):

    """
    [Joint 모델 역할]
    - 오디오 분리(Separation), 분류(Classification), ANC(Active Noise Control) 노이즈 저감을 한 번에 처리하는 end-to-end 모델 클래스
    - C-SudoRM-RF++ 기반 분리 → 안내방송음/그 외 소음 분류 → WaveNet-VNNs 기반 노이즈 저감(ANC) 일관 파이프라인 제공
    - 학습 및 추론 모두 지원하며, 저지연 처리를 염두에 둔 설계
    - 안내방송음/그 외 소음 분류를 통해 소스 자동 채널 매핑 가능
    """

    """
    Joint 모델의 모든 하위 블록(분리, 분류, ANC 저감)을 생성 및 사전학습된 가중치 로드
    환경설정(model_config)과 주요 파라미터(eta, clamp 범위 등) 저장, FIR 계수 불러오기 등 전체 시스템 상태 세팅
    분리모델, 저감모델, 분류기 등 블록별로 필요한 경로, 설정, checkpoint 등 모두 한 번에 등록
    불러오기 실패 시 fallback(더미값) 등 예외처리까지 포함
    """
    def __init__(self, sudormrf_checkpoint_path, wavenet_checkpoint_path, wavenet_config_path,
                 broadcast_classifier_checkpoint_path=None, use_broadcast_classifier=False, 
                 model_config=None):
        super().__init__()

        
        #설정값 관리
        self.model_config = model_config or {}
        self.eta_init_value = self.model_config.get('eta_init_value', 0.1)
        self.sef_clamp_range = self.model_config.get('sef_clamp_range', (-10.0, 10.0))
        self.classification_window_len = self.model_config.get('classification_window_len', 16000)
        self.audio_eps = self.model_config.get('audio_eps', 1e-9)

        from trainers.memory_manager import MemoryManager
        self.memory_manager = MemoryManager(self.model_config)
        self.use_broadcast_classifier = use_broadcast_classifier

        # C-SuDoRM-RF++ 분리 모델 초기화/가중치 로드  
        # 사전학습시 진행했던 하이퍼파라미터 구조 고정(실험에 따라 변경 가능)
        sudormrf_config = {
            'in_audio_channels': 1, 'out_channels': 256, 'in_channels': 384,
            'num_blocks': 16, 'upsampling_depth': 5, 'enc_kernel_size': 21,
            'enc_num_basis': 256, 'num_sources': 2
        }

        self.separation_model = CausalSuDORMRF(**sudormrf_config)
        self._load_sudormrf_weights(sudormrf_checkpoint_path)

        # WaveNet-VNNs ANC 저감 모델  
        # json config 기반 구조 생성 후 가중치 로드
        with open(wavenet_config_path, 'r') as f:
            wavenet_config = json.load(f)

        self.noise_reduction_model = WaveNet_VNNs(wavenet_config)
        self._load_wavenet_weights(wavenet_checkpoint_path)

        # BroadcastClassifier 안내방송음/그 외 소음 이진 분류기 추가
        if self.use_broadcast_classifier and BroadcastClassifier is not None:
            self.broadcast_classifier = BroadcastClassifier(window_len=16000)
            if broadcast_classifier_checkpoint_path and load_broadcast_classifier_weights:
                self.broadcast_classifier = load_broadcast_classifier_weights(
                    self.broadcast_classifier, broadcast_classifier_checkpoint_path
                )
            print("BroadcastClassifier added")
        else:
            self.broadcast_classifier = None

        # ANC 파라미터 설정
        self._setup_anc_paths()

        print("Joint model loaded")

    def _setup_anc_paths(self):
        """
        ANC(Active Noise Control) 파트에서 사용하는 FIR 필터(Primary/Secondary) 계수 및 eta 파라미터 등록
        mat 파일에서 계수 로드하며, 실패시 랜덤값으로 fallback
        eta: SEF(스피커 비선형 보정)용 파라미터
        """
        
        try:
            pri_path = "/Joint/channel/pri_channel.mat"
            sec_path = "/Joint/channel/sec_channel.mat"
            
            pri_channel = torch.tensor(
                loadmat(pri_path)["pri_channel"].squeeze(), 
                dtype=torch.float
            )
            sec_channel = torch.tensor(
                loadmat(sec_path)["sec_channel"].squeeze(), 
                dtype=torch.float
            )
            
            self.square_eta = self.model_config.get('eta_init_value', 0.1)
            
            self.register_buffer('pri_channel', pri_channel)
            self.register_buffer('sec_channel', sec_channel)
            self.register_buffer('pri_filter', pri_channel)
            self.register_buffer('sec_filter', sec_channel)
            self.eta = nn.Parameter(torch.tensor(self.square_eta))
            
        except Exception as e:
            print(f"Failed to load ANC paths: {e}")
            # Fallback: 더미 필터 생성
            filter_len = 64
            pri_channel = torch.randn(filter_len) * 0.01
            sec_channel = torch.randn(filter_len) * 0.01
            self.square_eta = self.model_config.get('eta_init_value', 0.1)
            
            self.register_buffer('pri_channel', pri_channel)
            self.register_buffer('sec_channel', sec_channel)
            self.register_buffer('pri_filter', pri_channel)
            self.register_buffer('sec_filter', sec_channel)
            self.eta = nn.Parameter(torch.tensor(self.square_eta))

    def _load_sudormrf_weights(self, checkpoint_path):
        """
        C-SudoRM-RF++ 모델 가중치 로드
        다양한 checkpoint 저장 포맷 지원(딕셔너리 key 자동 처리)
        Multi-GPU 저장시 key 클린업
        """
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in state:
            actual_state = state['model_state_dict']
        elif 'model' in state:
            actual_state = state['model']
        else:
            actual_state = state

        
        if any(key.startswith('module.') for key in actual_state.keys()):
            new_state = {key[7:] if key.startswith('module.') else key: value
                        for key, value in actual_state.items()}
            actual_state = new_state

        self.separation_model.load_state_dict(actual_state)

    def _load_wavenet_weights(self, checkpoint_path):
        """
        WaveNet-VNNs 저감 모델 가중치 로드
        """
        state = torch.load(checkpoint_path, map_location='cpu')
        model_state = state['model'] if 'model' in state else state
        self.noise_reduction_model.load_state_dict(model_state)

    def _sef_nonlinearity(self, signal, eta):
        """
        SEF 비선형 처리
        신호의 clip, NaN/Inf 보정 → SEF 적용 → 차원 변환 일관성 유지
        ANC 파이프라인에서 loudspeaker 비선형 대응 (WaveNet-VNNs 논문 방식)
        """        
        clamp_min, clamp_max = self.sef_clamp_range
        signal = torch.clamp(signal, clamp_min, clamp_max)

        if torch.isnan(signal).any() or torch.isinf(signal).any():
            signal = torch.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)        
        clamp_min, clamp_max = self.sef_clamp_range

        # SEF 함수 호출 전 차원 확인
        if signal.dim() == 3 and signal.shape[1] == 1:
            signal_for_sef = signal.squeeze(1)
        elif signal.dim() == 2:
            signal_for_sef = signal
        else:
            signal_for_sef = signal

        try:
            result = SEF(signal_for_sef, eta)
            
            if torch.isnan(result).any() or torch.isinf(result).any():
                result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if result.dim() == 2:
                result = result.unsqueeze(1)
            
            return result
            
        except Exception:
            # Fallback: 원본 신호 반환
            if signal_for_sef.dim() == 2:
                return signal_for_sef.unsqueeze(1)
            else:
                return signal_for_sef

    def preprocess_sep_chunk(self, wav_chunk, chunk_len):
        """
        분리모델 입력 전 음성데이터 전처리
        chunk 길이에 맞춰 zero-padding/자르기 + 평균0/표준편차1로 normalization
        """
        T = wav_chunk.shape[-1]
        if T < chunk_len:
            wav_chunk = F.pad(wav_chunk, (0, chunk_len - T), mode='constant')
        else:
            wav_chunk = wav_chunk[..., :chunk_len]

        m_mean = wav_chunk.mean(dim=-1, keepdim=True)
        m_std = wav_chunk.std(dim=-1, keepdim=True)
        normalized = (wav_chunk - m_mean) / (m_std + 1e-9)
        return normalized, m_mean, m_std, min(T, chunk_len)

    def postprocess_sep_chunk(self, est, m_mean, m_std, valid_len):
        """
        분리 결과를 원래 신호 스케일로 복원
        normalization 역변환 + padding/trim 보정
        """
        out = (est * (m_std + 1e-9) + m_mean)
        return out[..., :valid_len]

    def compute_sigmoid_mask(self, signal, classifier, device):
        """
        BroadcastClassifier 분류기를 활용한 방송/소음 마스킹  
        일정 구간(window_len) 단위로 자른 뒤, 각 segment별로 이진 분류기 확률값으로 mask 계산
        """
        window_len = self.classification_window_len
        L = signal.shape[-1]
        if classifier is None:
            return torch.zeros(signal.shape[-1], dtype=torch.float32, device=device)
            
        L = signal.shape[-1]
        masks = torch.zeros(L, dtype=torch.float32, device=device)

        sig_np = signal.squeeze().detach().cpu().numpy()

        for start in range(0, L, window_len):
            end = min(start + window_len, L)
            seg = sig_np[start:end]
            if seg.shape[0] < window_len:
                seg = np.pad(seg, (0, window_len - seg.shape[0]), mode="constant")

            x = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = classifier(x)
                prob_chunk = torch.sigmoid(logit).item()
            masks[start:end] = prob_chunk

        return masks

    def reduce_noise_like_inference(self, noise_signal, device):
        """
        WaveNet-VNNs 저감 모델로 ANC 처리
        노이즈 신호를 FIR → WaveNet → SEF → FIR 통과시켜 enhanced/antinoise 생성
        10초 단위 chunk로 잘라 메모리/속도 효율 높임
        """
        SEG_LEN = 10 * SR 
        N = noise_signal.shape[-1]
        outs_en, outs_dn = [], []

        with torch.no_grad():
            for start in range(0, N, SEG_LEN):
                end = min(start + SEG_LEN, N)
                seg = noise_signal[..., start:end]
                T = seg.shape[-1]

                tgt_gpu = fir_filter(self.pri_filter.to(device), seg)
                
                out = self.noise_reduction_model(seg)
                if out.dim() == 3:
                    out_flat = out.squeeze(1)
                elif out.dim() == 2:
                    out_flat = out
                else:
                    out_flat = out.unsqueeze(0)

                nl = SEF(out_flat, self.eta)
                dn = fir_filter(self.sec_filter.to(device), nl)
                en = tgt_gpu + dn

                outs_en.append(en[..., :T])
                outs_dn.append(dn[..., :T])

        enhanced = torch.cat(outs_en, dim=-1)
        anti_noise = torch.cat(outs_dn, dim=-1)
        return enhanced, anti_noise
    
    

    def _forward_direct_safe(self, mixed_input, chunk_len=None, return_classification=False):
        """
        학습 및 디버그에 특화된 forward 함수 (gradient 보장)
        입력 차원 자동보정, chunk 단위 분리→분류→ANC→합성까지 모든 동작을 통합 수행
        """
        device = mixed_input.device
        
        # 입력 차원 검증 및 수정
        if mixed_input.dim() == 4:
            mixed_input = mixed_input.squeeze(2)
        elif mixed_input.dim() == 2:
            mixed_input = mixed_input.unsqueeze(1)
        
        # 전체 길이와 청크 길이 확인
        total_length = mixed_input.shape[-1]
        
        if chunk_len is None:
            chunk_len = self.memory_manager.get_safe_chunk_size()
        
        # 청크 길이가 전체 길이보다 클 경우 조정
        if chunk_len > total_length:
            chunk_len = total_length
        
        # 분리 모델 처리
        with torch.amp.autocast('cuda', enabled=False):
            if chunk_len < total_length:
                # 청크별 분리 처리
                sep_chunks = []
                for start in range(0, total_length, chunk_len):
                    end = min(start + chunk_len, total_length)
                    chunk = mixed_input[:, :, start:end]
                    chunk_output = self.separation_model(chunk)
                    sep_chunks.append(chunk_output)
                
                sep_output = torch.cat(sep_chunks, dim=-1)
            else:
                sep_output = self.separation_model(mixed_input)
        
        # 2채널로 분리
        s1_clean = sep_output[:, 0:1, :]  # 안내방송음 채널
        s2_noise = sep_output[:, 1:2, :]  # 그 외 소음 채널

        classification_results = None
        broadcast_channel = s1_clean
        noise_channel = s2_noise

        if return_classification and self.broadcast_classifier is not None:
            try:
                with torch.amp.autocast('cuda', enabled=True):
                    # 1. 두 채널 모두 전처리
                    ch0_for_classification = self._prepare_for_classification(s1_clean)
                    ch1_for_classification = self._prepare_for_classification(s2_noise)
                    
                    # 2. 각각 분류 (원래 학습된 방식 그대로)
                    ch0_logits = self.broadcast_classifier(ch0_for_classification)  
                    ch1_logits = self.broadcast_classifier(ch1_for_classification)
                    
                    # 3. 확률 변환
                    ch0_prob = torch.sigmoid(ch0_logits).mean().item()  # 0~1 사이 확률
                    ch1_prob = torch.sigmoid(ch1_logits).mean().item()  # 0~1 사이 확률
                    
                    print(f"Ch0 (s1) broadcast prob: {ch0_prob:.3f}")
                    print(f"Ch1 (s2) broadcast prob: {ch1_prob:.3f}")
                    
                    # 4. 높은 확률을 가진 쪽이 안내방송음
                    if ch0_prob > ch1_prob:
                        print("Ch0 → Broadcast, Ch1 → Noise")
                        broadcast_channel = s1_clean  # ch0 = 안내방송음
                        noise_channel = s2_noise      # ch1 = 그 외 소음
                        is_ch0_broadcast = True
                    else:
                        print("Ch0 → Noise, Ch1 → Broadcast")
                        broadcast_channel = s2_noise  # ch1 = 안내방송음
                        noise_channel = s1_clean      # ch0 = 그 외 소음
                        is_ch0_broadcast = False
                    
                    # 5. 분류 결과 저장
                    classification_results = {
                        'ch0_prob': ch0_prob,
                        'ch1_prob': ch1_prob,
                        'ch0_logits': ch0_logits,
                        'ch1_logits': ch1_logits,
                        'is_ch0_broadcast': is_ch0_broadcast,
                        'confidence': abs(ch0_prob - ch1_prob)
                    }
                    
            except Exception as e:
                print(f"Classification failed: {e}")
                # 폴백: 기본 할당
                broadcast_channel = s1_clean
                noise_channel = s2_noise
                classification_results = None

        # WaveNet-VNNs ANC 처리
        with torch.amp.autocast('cuda', enabled=True):
            noise_input = noise_channel.squeeze(1)
            
            # device에 맞게 채널 이동
            pri_channel = self.pri_channel.to(device)
            sec_channel = self.sec_channel.to(device)
            
            # 메모리 절약을 위한 작은 청크 크기
            anc_chunk_len = min(16000, total_length // 4)
            
            if anc_chunk_len < total_length and total_length > 16000:
                # 청크별 ANC 처리
                chunks_target = []
                chunks_dn = []
                chunks_en = []
                
                for i in range(0, total_length, anc_chunk_len):
                    end_idx = min(i + anc_chunk_len, total_length)
                    actual_chunk_len = end_idx - i
                    
                    chunk = noise_input[:, i:end_idx]
                    
                    if chunk.shape[-1] > 0:
                        # 차원 변환: [B, T] -> [B, 1, T]
                        chunk_unsqueezed = chunk.unsqueeze(1)
                        
                        # FIR 필터 적용
                        target_chunk = fir_filter(pri_channel, chunk_unsqueezed)
                        
                        # WaveNet-VNNs 모델 호출
                        wavenet_output = self.noise_reduction_model(chunk_unsqueezed)

                        if torch.isnan(wavenet_output).any() or torch.isinf(wavenet_output).any():
                            wavenet_output = torch.nan_to_num(wavenet_output, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # WaveNet-VNNs 출력 차원 정규화
                        if wavenet_output.dim() == 3 and wavenet_output.shape[1] == 1:
                            wavenet_flat = wavenet_output.squeeze(1)
                        elif wavenet_output.dim() == 2:
                            wavenet_flat = wavenet_output
                        elif wavenet_output.dim() == 1:
                            wavenet_flat = wavenet_output.unsqueeze(0)
                        else:
                            wavenet_flat = wavenet_output.view(chunk.shape[0], -1)
                        
                        # SEF 비선형 처리
                        nonlinear_chunk = self._sef_nonlinearity(wavenet_flat, self.square_eta)
                        
                        # fir_filter를 위해 3차원으로 변환
                        if nonlinear_chunk.dim() == 2:
                            nonlinear_for_filter = nonlinear_chunk.unsqueeze(1)
                        elif nonlinear_chunk.dim() == 3:
                            nonlinear_for_filter = nonlinear_chunk
                        else:
                            nonlinear_for_filter = nonlinear_chunk.view(chunk.shape[0], 1, -1)
                        
                        # FIR 필터 적용
                        dn_chunk = fir_filter(sec_channel, nonlinear_for_filter)
                        
                        # WaveNet 스타일 합성
                        en_chunk = dn_chunk + target_chunk
                        
                        # 크기 검증 및 수정 함수
                        def fix_chunk_shape(tensor, expected_shape):
                            if tensor.shape != expected_shape:
                                # 배치 차원 수정
                                if tensor.shape[0] != expected_shape[0]:
                                    if tensor.shape[0] == expected_shape[2]:
                                        tensor = tensor.permute(2, 1, 0) if tensor.dim() == 3 else tensor.permute(1, 0)
                                        if tensor.dim() == 2:
                                            tensor = tensor.unsqueeze(1)
                                    else:
                                        tensor = tensor.view(expected_shape)
                                
                                # 채널 차원 수정
                                if tensor.dim() == 2:
                                    tensor = tensor.unsqueeze(1)
                                elif tensor.dim() == 3 and tensor.shape[1] != 1:
                                    tensor = tensor[:, :1, :]
                                
                                # 길이 차원 수정
                                if tensor.shape[-1] != expected_shape[-1]:
                                    tensor = tensor[..., :expected_shape[-1]]
                                
                                # 최종 크기 확인
                                if tensor.shape != expected_shape:
                                    tensor = tensor.view(expected_shape)
                            
                            return tensor
                        
                        expected_shape = (chunk.shape[0], 1, actual_chunk_len)
                        target_chunk = fix_chunk_shape(target_chunk, expected_shape)
                        dn_chunk = fix_chunk_shape(dn_chunk, expected_shape)
                        en_chunk = fix_chunk_shape(en_chunk, expected_shape)
                        
                        chunks_target.append(target_chunk)
                        chunks_dn.append(dn_chunk)
                        chunks_en.append(en_chunk)
                        
                        # 메모리 정리
                        del chunk_unsqueezed, target_chunk, wavenet_output, wavenet_flat
                        del nonlinear_chunk, nonlinear_for_filter, dn_chunk, en_chunk
                        torch.cuda.empty_cache()
                
                if chunks_target:
                    try:
                        # 메모리 절약 연결
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                            
                            s2_target = torch.cat(chunks_target, dim=-1)
                            del chunks_target
                            torch.cuda.empty_cache()
                            
                            s2_antinoise = torch.cat(chunks_dn, dim=-1)
                            del chunks_dn
                            torch.cuda.empty_cache()
                            
                            s2_enhanced = torch.cat(chunks_en, dim=-1)
                            del chunks_en
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        # 폴백: 첫 번째 chunk만 사용
                        first_target = chunks_target[0] if chunks_target else torch.zeros(1, 1, anc_chunk_len, device=device)
                        first_dn = chunks_dn[0] if chunks_dn else torch.zeros(1, 1, anc_chunk_len, device=device)
                        first_en = chunks_en[0] if chunks_en else torch.zeros(1, 1, anc_chunk_len, device=device)
                        
                        remaining_len = total_length - first_target.shape[-1]
                        if remaining_len > 0:
                            zero_pad = torch.zeros(1, 1, remaining_len, device=device)
                            s2_target = torch.cat([first_target, zero_pad], dim=-1)
                            s2_antinoise = torch.cat([first_dn, zero_pad], dim=-1)
                            s2_enhanced = torch.cat([first_en, zero_pad], dim=-1)
                        else:
                            s2_target = first_target
                            s2_antinoise = first_dn
                            s2_enhanced = first_en
                        
                        # 메모리 정리
                        del chunks_target, chunks_dn, chunks_en
                        torch.cuda.empty_cache()
                else:
                    s2_target = torch.zeros_like(s2_noise)
                    s2_antinoise = torch.zeros_like(s2_noise)
                    s2_enhanced = torch.zeros_like(s2_noise)
            else:
                # 전체 길이 ANC 처리
                try:
                    noise_unsqueezed = noise_input.unsqueeze(1)
                    
                    s2_target = fir_filter(pri_channel, noise_unsqueezed)
                    
                    wavenet_output = self.noise_reduction_model(noise_unsqueezed)
                    
                    # WaveNet-VNNs 출력 차원 조정
                    if wavenet_output.dim() == 3:
                        wavenet_flat = wavenet_output.squeeze(1)
                    elif wavenet_output.dim() == 2:
                        wavenet_flat = wavenet_output
                    else:
                        wavenet_flat = wavenet_output.view(wavenet_output.shape[0], -1)
                    
                    # SEF 비선형
                    nonlinear_out = self._sef_nonlinearity(wavenet_flat, self.square_eta)
                    
                    # fir_filter용 차원 확인
                    if nonlinear_out.dim() == 2:
                        nonlinear_for_filter = nonlinear_out.unsqueeze(1)
                    elif nonlinear_out.dim() == 3:
                        nonlinear_for_filter = nonlinear_out
                    else:
                        nonlinear_for_filter = nonlinear_out
                    
                    s2_antinoise = fir_filter(sec_channel, nonlinear_for_filter)
                    
                    s2_enhanced = s2_antinoise + s2_target
                    
                except Exception:
                    s2_target = torch.zeros_like(s2_noise)
                    s2_antinoise = torch.zeros_like(s2_noise)
                    s2_enhanced = torch.zeros_like(s2_noise)
        
        # 최종 재합성
        if hasattr(self, 'eta'):
            eta_weight = torch.sigmoid(self.eta)
            enhanced_verification = eta_weight * broadcast_channel + (1 - eta_weight) * s2_enhanced
        else:
            enhanced_verification = 0.7 * broadcast_channel + 0.3 * s2_enhanced
            
        return {
            's1_clean': broadcast_channel,
            's2_noise': noise_channel,
            's2_target': s2_target,
            's2_antinoise': s2_antinoise,
            's2_enhanced': s2_enhanced,
            'enhanced_verification': enhanced_verification,
            'classification': classification_results
        }

    def forward_inference_style(self, mixed_input, chunk_len=None, return_classification=False):
        """
          실전 추론/배치 테스트용 forward 함수(gradient X)
          입력 전체를 chunk 단위로 분리/후처리, 분류기로 채널 역할 할당, ANC 적용, 최종 결과 합성
        """
        
        batch_size = mixed_input.shape[0]
        L = mixed_input.shape[-1]
        device = mixed_input.device

        # 적응적 청크 크기 결정
        if chunk_len is None:
            chunk_len = self.memory_manager.get_safe_chunk_size()

        # 입력 차원 표준화
        mixed_input = standardize_audio_dims(mixed_input)

        # 1단계: 분리(Separation)
        max_num_sources = 2
        sep_acc = torch.zeros((batch_size, max_num_sources, L), dtype=torch.float32, device=device)

        for start in range(0, L, chunk_len):
            end = min(start + chunk_len, L)
            chunk = mixed_input[..., start:end]
            
            # 전처리 (배치 단위)
            mix_t, m_mean, m_std, valid = self.preprocess_sep_chunk(chunk, chunk_len)
            
            with torch.no_grad():
                est = self.separation_model(mix_t)
                est = mixture_consistency.apply(est, mix_t)
            
            # 후처리 (배치 단위)
            sep_chunk = self.postprocess_sep_chunk(est, m_mean, m_std, valid)
            sep_acc[:, :, start:start+valid] = sep_chunk

        # 2단계: 분류 및 채널 할당
        classification_results = {}
        
        if self.use_broadcast_classifier and self.broadcast_classifier is not None:
            # 배치 단위로 분류 수행
            batch_broadcast_channels = []
            batch_noise_channels = []
            batch_classification_info = []
            
            for b in range(batch_size):
                # 각 샘플의 두 채널을 _prepare_for_classification 방식으로 분류
                ch0_tensor = sep_acc[b, 0:1].unsqueeze(0)  # [1, 1, T]
                ch1_tensor = sep_acc[b, 1:2].unsqueeze(0)  # [1, 1, T]
                
                # 분류기용 전처리
                ch0_prep = self._prepare_for_classification(ch0_tensor)  # [1, 1, 16000]
                ch1_prep = self._prepare_for_classification(ch1_tensor)  # [1, 1, 16000]
                
                # 각 채널의 방송 확률 계산
                with torch.no_grad():
                    ch0_logits = self.broadcast_classifier(ch0_prep)
                    ch1_logits = self.broadcast_classifier(ch1_prep)
                    
                    ch0_prob = torch.sigmoid(ch0_logits).mean().item()
                    ch1_prob = torch.sigmoid(ch1_logits).mean().item()
                
                print(f"Sample {b}: Ch0 prob={ch0_prob:.3f}, Ch1 prob={ch1_prob:.3f}")
                
                # 확률 비교로 안내방송음/그 외 소음 채널 결정
                if ch0_prob > ch1_prob:
                    print(f"Sample {b}: Ch0 → Broadcast, Ch1 → Noise")
                    broadcast_channel = sep_acc[b, 0]  # 채널0이 안내방송음
                    noise_channel = sep_acc[b, 1]      # 채널1이 그 외 소음
                    is_channel0_broadcast = True
                else:
                    print(f"Sample {b}: Ch0 → Noise, Ch1 → Broadcast")
                    broadcast_channel = sep_acc[b, 1]  # 채널1이 안내방송음
                    noise_channel = sep_acc[b, 0]      # 채널0이 그 외 소음
                    is_channel0_broadcast = False
                
                batch_broadcast_channels.append(broadcast_channel)
                batch_noise_channels.append(noise_channel)
                
                # 분류 정보 저장
                batch_classification_info.append({
                    'ch0_prob': ch0_prob,
                    'ch1_prob': ch1_prob,
                    'is_channel0_broadcast': is_channel0_broadcast,
                    'confidence': abs(ch0_prob - ch1_prob)
                })
            
            # 배치로 재구성
            sep_broadcast = torch.stack(batch_broadcast_channels).unsqueeze(1)  
            sep_noise = torch.stack(batch_noise_channels).unsqueeze(1)        
            
            classification_results = {
                'batch_info': batch_classification_info,
                'average_accuracy': None
            }
            
            print(f"Classification completed for {batch_size} samples")
            
        else:
            print("No classifier available, using default assignment")
            # 만약 분류기가 없으면 기본적으로 첫 번째 채널을 안내방송음으로 가정
            sep_broadcast = sep_acc[:, 0:1]  
            sep_noise = sep_acc[:, 1:2]      

        # 3단계: WaveNet-VNNs(ANC) - 분류된 그 외 소음 채널에 적용
        print(f"Applying ANC to classified noise channels...")
        batch_enhanced = []
        batch_anti_noise = []
        
        for b in range(batch_size):
            # 분류 결과에 따른 그 외 소음 채널 사용
            noise_sample = sep_noise[b:b+1]  # 이제 분류된 그 외 소음 채널
            enhanced_sample, anti_sample = self.reduce_noise_like_inference(noise_sample, device)
            batch_enhanced.append(enhanced_sample)
            batch_anti_noise.append(anti_sample)
        
        # 배치로 재구성
        s2_enhanced = torch.cat(batch_enhanced, dim=0)    
        s2_antinoise = torch.cat(batch_anti_noise, dim=0)

        # 4단계: 재합성 - 분류된 안내방송음 채널 + ANC 처리된 노이즈
        print(f"Final mixing: classified broadcast + enhanced noise")
        final_mix = sep_broadcast + s2_enhanced

        # 결과 반환
        results = {
            's1_clean': sep_broadcast,        # 분류된 안내방송음 채널
            's2_noise': sep_noise,            # 분류된 그 외 소음 채널
            's2_target': None,
            's2_antinoise': s2_antinoise,
            's2_enhanced': s2_enhanced,
            'enhanced_verification': final_mix,
            'sep_acc': sep_acc
        }

        # s2_target 계산 (분류된 그 외 소음 채널 기반)
        results['s2_target'] = fir_filter(self.pri_filter.to(device), sep_noise)

        # 분류 결과 추가
        if return_classification and classification_results:
            results['classification'] = classification_results

        return results
    
    def _prepare_for_classification(self, audio):
        """
        BroadcastClassifier 분류기 입력 포맷으로 shape, 길이 변환 및 16,000길이로 패딩/자르기
        모든 분류 input이 동일 shape 유지하도록 보장
        """
        # 3차원으로 만들기
        while audio.dim() > 3:
            min_dim_idx = audio.shape.index(min(audio.shape[1:-1]))
            audio = audio.squeeze(min_dim_idx + 1)
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        if audio.dim() != 3:
            raise ValueError(f"Expected 3D tensor for classifier, got {audio.dim()}D")
        
        # 길이 조정 (BroadcastClassifier는 16000 길이 기대)
        current_length = audio.shape[-1]
        if current_length > 16000:
            start_idx = (current_length - 16000) // 2            
            audio = audio[..., start_idx:start_idx + 16000]
        elif current_length < 16000:
            pad_len = 16000 - current_length
            audio = F.pad(audio, (0, pad_len), mode='constant', value=0)
        
        return audio

    def forward_for_training(self, mixed_input, chunk_len=None, return_classification=False):
        """
        학습용 forward 함수
        내부적으로 _forward_direct_safe 호출
        """
        return self._forward_direct_safe(mixed_input, chunk_len, return_classification)

    def forward(self, mixed_input, chunk_len=None, return_classification=False):
        """
        PyTorch 모델 forward 메인 진입점
        self.training에 여부 따라 자동으로 학습/추론 파이프라인 분기
        """
        if self.training:
            return self.forward_for_training(mixed_input, chunk_len, return_classification)
        else:
            return self.forward_inference_style(mixed_input, chunk_len, return_classification)

    def get_trainer_compatible_params(self):
        """
        외부 trainer/스크립트와 연동을 위한 핵심 파라미터 반환  
        FIR 계수 및 eta 값 반환(ANC 파이프라인 하드웨어 연동시 사용)
        """
        return {
            'pri_channel': self.pri_channel,
            'sec_channel': self.sec_channel,
            'square_eta': self.square_eta
        }