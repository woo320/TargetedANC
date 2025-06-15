import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

# SudoRM-RF PIT 손실함수 import
try:
    from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
    ASTEROID_AVAILABLE = True
except ImportError:
    ASTEROID_AVAILABLE = False
    print("⚠️ Asteroid library not available - using basic SI-SDR")

from models.model_utils import compute_broadcast_classification_loss
# WaveNet 경로 추가
sys.path.insert(0, "/content/drive/MyDrive/joint/WaveNet-VNNs-for-ANC/WaveNet_VNNs")
from loss_function import dBA_Loss, NMSE_Loss
from utils import fir_filter, SEF

# 로컬 모듈 import
from datasets.sudormrf_dataset import SudoRMRFDynamicMixDataset
from datasets.improved_dataset import ImprovedAudioDataset
from datasets.collate_functions import sudormrf_dynamic_mix_collate_fn, improved_collate_fn
from .memory_manager import MemoryManager
from project_utils import EarlyStoppingManager
from project_utils.audio_utils import standardize_audio_dims
from project_utils.augmentation import online_augment_sudormrf
from config.constants import SR

class ImprovedJointTrainerWithSudoRMRFMix:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_manager = MemoryManager(config)
        self.accumulation_steps = config.get('accumulation_steps', 4)
        
        # SudoRM-RF 설정
        self.use_online_augment = config.get('use_online_augment', True)
        self.use_pit_loss = config.get('use_pit_loss', True) and ASTEROID_AVAILABLE

        self._setup_paths()
        self._setup_model()
        self._setup_dataloaders()
        self._setup_anc_paths()
        self._setup_training()

        if config.get('use_tensorboard', True):
            self.writer = SummaryWriter(self.log_path)

        pit_status = "with PIT loss" if self.use_pit_loss else "with basic SI-SDR"
        print(f"Joint Trainer Mixed Training (Training:Dynamic + Val:Premixed) {pit_status} initialized")

    def _setup_paths(self):
        timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
        mix_suffix = "_mixed_training"  # Training(동적) + Validation(premixed)
        self.exp_path = f"/content/drive/MyDrive/joint/result/joint{mix_suffix}_{timestamp}"
        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'weights')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def _setup_model(self):
        from models.joint_model import ImprovedJointModel
        self.model = ImprovedJointModel(
            sudormrf_checkpoint_path=self.config['sudormrf_checkpoint'],
            wavenet_checkpoint_path=self.config['wavenet_checkpoint'],
            wavenet_config_path=self.config['wavenet_config'],
            broadcast_classifier_checkpoint_path=self.config.get('broadcast_classifier_checkpoint'),
            use_broadcast_classifier=self.config.get('use_broadcast_classifier', False),
            model_config=self.config
        ).to(self.device)

        # 학습 가능한 파라미터 확인
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model loaded with {trainable_params:,} trainable parameters")

    def _setup_dataloaders(self):
        """데이터로더 설정 - config를 collate_fn에 전달"""
        dataloaders = {}

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.config['dataset_root'], split)

            if not os.path.exists(split_dir):
                dataloaders[split] = None
                continue

            try:
                # 샘플 수 제한
                max_samples = None
                if split == 'train' and self.config.get('limit_train_samples'):
                    max_samples = self.config['limit_train_samples']
                elif split == 'val' and self.config.get('limit_val_samples'):                    
                    max_samples = self.config['limit_val_samples']
                elif split == 'test' and self.config.get('limit_test_samples'):
                    max_samples = self.config['limit_test_samples']

                # split별로 명확히 구분
                if split == 'train':
                    # Training: 무조건 동적 믹스
                    dataset = SudoRMRFDynamicMixDataset(
                        self.config['dataset_root'],
                        split=split,
                        max_samples=max_samples,
                        max_duration=self.config.get('max_audio_duration', 15.0),
                        use_online_augment=self.use_online_augment
                    )
                    collate_fn = lambda batch: sudormrf_dynamic_mix_collate_fn(batch, self.config)
                    
                else:
                    # Validation/Test: 무조건 pre-mixed
                    dataset = ImprovedAudioDataset(
                        self.config['dataset_root'],
                        split=split,
                        max_samples=max_samples,
                        max_duration=self.config.get('max_audio_duration', 15.0)
                    )
                    collate_fn = lambda batch: improved_collate_fn(batch, self.config)

                # 적응적 배치 크기
                batch_size = self.memory_manager.adaptive_batch_size(
                    self.config.get('batch_size', 1)
                )

                if split in ['validation', 'test']:
                    batch_size = min(batch_size * 2, self.config.get('batch_size', 1) * 2)

                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    num_workers=self.config.get('dataloader_num_workers', 0),
                    pin_memory=self.config.get('dataloader_pin_memory', False),
                    drop_last=(split == 'train' and self.config.get('dataloader_drop_last', True)),
                    collate_fn=collate_fn
                )

                dataloaders[split] = dataloader

            except Exception as e:
                print(f"Failed to create {split} dataloader: {e}")
                dataloaders[split] = None

        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']

    def _setup_anc_paths(self):
        params = self.model.get_trainer_compatible_params()
        self.pri_channel = params['pri_channel']
        self.sec_channel = params['sec_channel'] 
        self.square_eta = params['square_eta']

    def _setup_training(self):
        self.model.train()
        
        # 파라미터 그룹 설정
        separation_params = list(self.model.separation_model.parameters())
        noise_params = list(self.model.noise_reduction_model.parameters())
        eta_params = [self.model.eta] if hasattr(self.model, 'eta') else []

        classifier_params = []
        if hasattr(self.model, 'broadcast_classifier') and self.model.broadcast_classifier is not None:
            classifier_params = list(self.model.broadcast_classifier.parameters())

        param_groups = [
            {'params': separation_params, 'lr': self.config['separation_lr']},
            {'params': noise_params, 'lr': self.config['noise_lr']},
        ]
        if eta_params:
            param_groups.append({'params': eta_params, 'lr': self.config['eta_lr']})
        
        if classifier_params:
            param_groups.append({
                'params': classifier_params, 
                'lr': self.config['classifier_lr']
            })

        self.optimizer = optim.Adam(param_groups)

        # 수정된 스케줄러 설정
        scheduler_params = {
            'mode': 'min',
            'factor': 0.8,                     # 0.7 → 0.8 (덜 급격하게)
            'patience': 4,                     # 2 → 4 (더 인내심 있게, 조인트 학습 고려)
            'min_lr': 5e-8,                    # 1e-7 → 5e-8 (더 낮은 최소값)
            'threshold': 0.001,                # 새로 추가: 의미있는 개선만 인정
            'threshold_mode': 'rel'            # 상대적 개선
        }
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_params
        )

        # 손실 함수들
        self.bce_loss = torch.nn.BCEWithLogitsLoss().to(self.device)
        
        # SudoRM-RF PIT 손실함수
        if self.use_pit_loss and ASTEROID_AVAILABLE:
            self.sudormrf_loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        else:
            self.sudormrf_loss_func = None
        
        self.best_val_loss = float('inf')

        # 수정된 조기 종료 설정
        if 'early_stopping_patience' not in self.config or 'min_delta' not in self.config:
            raise ValueError("early_stopping_patience와 min_delta는 config에서 명시적으로 설정해야 합니다!")

        self.early_stopping = EarlyStoppingManager(
            patience=self.config['early_stopping_patience'],     
            min_delta=self.config['min_delta'],
            metric_weights=self.config.get('early_stopping_weights', {
                'anc_total': 0.4,
                'separation_loss': 0.3, 
                'classification_accuracy': 0.2,
                'final_quality': 0.1
            })
)

    def _sisdr_loss_basic(self, est, target, zero_mean=True, eps=1e-9):
        """기본 SI-SDR loss"""
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

    def _compute_separation_loss_sudormrf_style(self, estimated_sources, target_sources):
        """SudoRM-RF 스타일 분리 손실 계산"""
        # 차원 불일치 해결
        if estimated_sources.shape != target_sources.shape:
            min_batch = min(estimated_sources.shape[0], target_sources.shape[0])
            estimated_sources = estimated_sources[:min_batch]
            target_sources = target_sources[:min_batch]
            
            min_time = min(estimated_sources.shape[-1], target_sources.shape[-1])
            estimated_sources = estimated_sources[..., :min_time]
            target_sources = target_sources[..., :min_time]
        
        if self.use_pit_loss and self.sudormrf_loss_func is not None:
            try:
                if (estimated_sources.shape == target_sources.shape and 
                    estimated_sources.requires_grad and 
                    target_sources.dtype == estimated_sources.dtype and
                    len(estimated_sources.shape) == 3 and
                    estimated_sources.shape[1] == 2):
                    
                    pit_loss = self.sudormrf_loss_func(estimated_sources, target_sources)
                    
                    if torch.isnan(pit_loss) or torch.isinf(pit_loss):
                        raise ValueError("Invalid PIT loss")
                    
                    return pit_loss
                else:
                    raise ValueError("PIT requirements not met")
                    
            except Exception:
                # Fallback to basic SI-SDR
                s1_loss = self._sisdr_loss_basic(estimated_sources[:, 0], target_sources[:, 0])
                s2_loss = self._sisdr_loss_basic(estimated_sources[:, 1], target_sources[:, 1])
                return (s1_loss + s2_loss) / 2
        else:
            # 기본 SI-SDR 사용
            s1_loss = self._sisdr_loss_basic(estimated_sources[:, 0], target_sources[:, 0])
            s2_loss = self._sisdr_loss_basic(estimated_sources[:, 1], target_sources[:, 1])
            return (s1_loss + s2_loss) / 2 
        
    def safe_dba_loss(self, signal, target=None, eps=1e-8):
        """추론코드와 동일한 dBA 계산 (차이 방식)"""
        try:
            if target is not None:
                # 추론코드와 동일: dBA(enhanced) - dBA(target)
                signal_power = torch.mean(signal ** 2)
                target_power = torch.mean(target ** 2)
                
                signal_db = 10 * torch.log10(signal_power + eps)
                target_db = 10 * torch.log10(target_power + eps)
                
                dba_diff = signal_db - target_db
                return dba_diff
            else:
                # 단일 신호의 파워
                power = torch.mean(signal ** 2)
                db_power = 10 * torch.log10(power + eps)
                return db_power
                    
        except Exception:
            if target is not None:
                return torch.mean((signal - target) ** 2)
            else:
                return torch.mean(signal ** 2)

    def safe_nmse_loss(self, signal, target, eps=1e-8):
        """수정된 NMSE 계산"""
        try:
            # 차원 확인
            if signal.dim() != target.dim():
                signal = signal.view(-1)
                target = target.view(-1)
            
            # 길이 맞추기
            min_len = min(signal.shape[-1], target.shape[-1])
            signal = signal[..., :min_len]
            target = target[..., :min_len]
            
            # 정확한 NMSE 계산
            signal_power = torch.sum(signal ** 2) + eps
            target_power = torch.sum(target ** 2) + eps
            
            # 0으로 나누기 방지
            if target_power < eps * 10:
                return torch.mean((signal - target) ** 2)
            
            nmse_linear = signal_power / target_power
            nmse_db = 10 * torch.log10(nmse_linear + eps)
            
            # 범위 제한 (-50dB ~ +50dB)
            nmse_db = torch.clamp(nmse_db, -30.0, 30.0)
            
            return nmse_db
            
        except Exception as e:
            print(f"NMSE calculation failed: {e}")
            return torch.mean((signal - target) ** 2)
    
    def _compute_losses(self, outputs, batch):
        # 입력 데이터 처리
        if 'input' in batch:
            input_mixed = batch['input'].to(self.device)
        elif 'sources' in batch:
            sources = batch['sources'].to(self.device)
            input_mixed = sources.sum(dim=1, keepdim=True)
        else:
            raise ValueError("No valid input found in batch")

        target_s1 = batch['separation_targets']['s1'].to(self.device)
        target_s2 = batch['separation_targets']['s2'].to(self.device)
        
        # 차원 표준화
        input_mixed = standardize_audio_dims(input_mixed)
        target_s1 = standardize_audio_dims(target_s1)
        target_s2 = standardize_audio_dims(target_s2)
        
        # 배치 크기 맞추기
        batch_sizes = [input_mixed.shape[0], target_s1.shape[0], target_s2.shape[0]]
        min_batch_size = min(batch_sizes)
        
        if len(set(batch_sizes)) > 1:
            input_mixed = input_mixed[:min_batch_size]
            target_s1 = target_s1[:min_batch_size]
            target_s2 = target_s2[:min_batch_size]

        # 모델 출력
        s1_clean = outputs['s1_clean']
        s2_noise = outputs['s2_noise']
        s2_antinoise = outputs['s2_antinoise']
        s2_enhanced = outputs['s2_enhanced']
        enhanced_verification = outputs['enhanced_verification']

        # 분류 손실 계산
        classification_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        classification_accuracy = 0.0
        accuracy_weight = 1.0
        
        if 'classification' in outputs and outputs['classification'] is not None:
            try:
                classification_results = outputs['classification']
                
                if isinstance(classification_results, torch.Tensor):
                    # 타겟 생성: s1과 s2의 에너지 비교
                    s1_energy = torch.mean(s1_clean.abs(), dim=-1)
                    s2_energy = torch.mean(s2_noise.abs(), dim=-1)
                    targets = (s1_energy > s2_energy).float().squeeze(-1)
                    
                    classification_loss = self.bce_loss(
                        classification_results.squeeze(-1) if classification_results.dim() > 1 else classification_results,
                        targets
                    )

                    probs = torch.sigmoid(classification_results.squeeze())
                    predicted = (probs > 0.5).float()
                    classification_accuracy = (predicted == targets).float().mean().item()
                
                accuracy_weight = 1.0 + (1.0 - classification_accuracy) * 0.5
                
            except Exception:
                classification_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                classification_accuracy = 0.0
                accuracy_weight = 1.0
        
        # batch 타입으로 Training/Validation 구분
        if 'sources' in batch:
            # Training: SudoRM-RF 스타일 PIT 손실
            estimated_sources = torch.stack([
                s1_clean.squeeze(1),
                s2_noise.squeeze(1)
            ], dim=1)
            
            target_sources = torch.stack([
                target_s1.squeeze(1),
                target_s2.squeeze(1)
            ], dim=1).float()
            
            separation_loss = self._compute_separation_loss_sudormrf_style(
                estimated_sources, target_sources
            ) * accuracy_weight
            
            s1_separation_loss = separation_loss / 2  # 로깅용 근사값
            s2_separation_loss = separation_loss / 2  # 로깅용 근사값
            separation_method = "PIT-SISDR" if self.use_pit_loss else "SI-SDR Average"
            
        else:
            # Validation/Test: 개별 SI-SDR 계산
            s1_separation_loss = self._sisdr_loss_basic(s1_clean, target_s1) * accuracy_weight
            s2_separation_loss = self._sisdr_loss_basic(s2_noise, target_s2) * accuracy_weight
            separation_loss = (s1_separation_loss + s2_separation_loss) / 2
            separation_method = "Individual SI-SDR"

        # 안전한 ANC 손실 계산 (추론 방식으로 통일)
        try:
            target_flat = outputs['s2_target'].squeeze(1)
            en_flat = outputs['s2_enhanced'].squeeze(1)
            
            # dBA: enhanced_dB - target_dB
            anc_dba_loss = self.safe_dba_loss(en_flat, target_flat)
            
            # NMSE: 10 * log10(enhanced_power / target_power)
            anc_nmse_loss = self.safe_nmse_loss(en_flat, target_flat)
            
            # 계산 후에 디버그 출력 (수정됨)
            debug_prob = self.config.get('debug_loss_print_prob', 0.05)
            if torch.rand(1).item() < debug_prob:
                print(f"ANC Debug: dBA={anc_dba_loss.item():.4f}dB, "
                        f"NMSE={anc_nmse_loss.item():.4f}dB, "
                        f"Signal_std={en_flat.std().item():.6f}, "
                        f"Target_std={target_flat.std().item():.6f}")
            
            # NaN/Inf 검증
            if torch.isnan(anc_dba_loss) or torch.isinf(anc_dba_loss):
                anc_dba_loss = torch.mean((en_flat - target_flat) ** 2)
            
            if torch.isnan(anc_nmse_loss) or torch.isinf(anc_nmse_loss):
                anc_nmse_loss = torch.mean((en_flat - target_flat) ** 2)
            
            # Gradient 보장
            if not anc_dba_loss.requires_grad:
                anc_dba_loss = anc_dba_loss.requires_grad_(True)
            if not anc_nmse_loss.requires_grad:
                anc_nmse_loss = anc_nmse_loss.requires_grad_(True)
                
        except Exception as e:
            print(f"ANC loss calculation failed: {e}")
            # 기본값으로 설정
            anc_dba_loss = torch.mean((en_flat - target_flat) ** 2).requires_grad_(True)
            anc_nmse_loss = torch.mean((en_flat - target_flat) ** 2).requires_grad_(True)

        # 최종 품질 손실
        final_quality_loss = self._sisdr_loss_basic(enhanced_verification, target_s1)

        # 안티노이즈 제약
        antinoise_magnitude = torch.mean(torch.abs(s2_antinoise))
        target_antinoise_magnitude = self.config.get('target_antinoise_magnitude', 0.1)
        antinoise_constraint = torch.abs(antinoise_magnitude - target_antinoise_magnitude)

        # ANC 총 손실
        anc_weights = self.config.get('anc_loss_weights', {
            'dba_weight': 0.5,
            'nmse_weight': 0.5
        })
        anc_total_loss = (anc_weights['dba_weight'] * anc_dba_loss + 
                         anc_weights['nmse_weight'] * anc_nmse_loss)

        # 🔧 간단한 총 손실 계산
        loss_weights = self.config.get('loss_weights', {
            'final_quality': 0.12,
            'anc_total': 0.28,
            'separation': 0.42,
            'classification': 0.15,
            'antinoise_constraint': 0.03
        })
        
        # 모든 경우에 동일한 가중치 적용
        total_loss = (
            loss_weights['final_quality'] * final_quality_loss +
            loss_weights['anc_total'] * anc_total_loss +
            loss_weights['separation'] * separation_loss +
            loss_weights['classification'] * classification_loss +
            loss_weights['antinoise_constraint'] * antinoise_constraint
        )

        # 최종 손실 검증
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = final_quality_loss
            
        if not total_loss.requires_grad:
            print("Total loss does not require gradients!")

        return {
            'total_loss': total_loss,
            'anc_dba_loss': anc_dba_loss,
            'anc_nmse_loss': anc_nmse_loss,
            'anc_total_loss': anc_total_loss,
            'final_quality_loss': final_quality_loss,
            's1_separation_loss': s1_separation_loss,
            's2_separation_loss': s2_separation_loss,
            'separation_loss': separation_loss,
            'antinoise_constraint': antinoise_constraint,
            'classification_loss': classification_loss,
            'classification_accuracy': classification_accuracy,
            'use_pit': self.use_pit_loss,
            'separation_method': separation_method
        }

    def _train_epoch(self, epoch):
        """훈련 에포크 (수정된 손실 로깅)"""
        self.model.train()

        total_losses = {
            'total': 0, 'anc_dba': 0, 'anc_nmse': 0, 'anc_total': 0,
            'final_quality': 0, 's1_separation': 0, 's2_separation': 0,
            'antinoise_constraint': 0, 'classification': 0
        }
        
        classification_accuracies = []
        aug_stats = []  # Training에서만 사용

        num_batches = len(self.train_loader)
        pit_status = " (PIT)" if self.use_pit_loss else " (Basic)"
        train_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}{pit_status}', ncols=150)

        self.optimizer.zero_grad()

        for step, batch in enumerate(train_bar, 1):
            # 메모리 모니터링
            allocated, reserved = self.memory_manager.get_memory_usage()

            try:
                memory_warning_threshold = self.config.get('memory_warning_threshold', 20.0)
                if step % 20 == 0 and allocated > memory_warning_threshold:
                    self.memory_manager.cleanup_memory(aggressive=True)

                # 데이터 전처리
                if 'sources' in batch:
                    # Training: 동적 믹스
                    sources = batch['sources'].to(self.device)

                    if self.use_online_augment:
                        sources_aug = online_augment_sudormrf(sources)
                        aug_ratio = torch.mean(sources_aug / (sources + 1e-9)).item()
                        aug_stats.append(aug_ratio)
                    else:
                        sources_aug = sources

                    mixed_input = sources_aug.sum(dim=1, keepdim=True)
                    
                    m = mixed_input.mean(dim=-1, keepdim=True)
                    s = mixed_input.std(dim=-1, keepdim=True)
                    mixed_input = (mixed_input - m) / (s + 1e-9)
                else:
                    # Validation: premixed
                    mixed_input = batch['input'].to(self.device)

                mixed_input.requires_grad_(True)

                # 적응적 청크 크기
                adaptive_chunk = self.memory_manager.get_safe_chunk_size()

                # 모델 forward
                outputs = self.model.forward_for_training(
                    mixed_input, 
                    chunk_len=adaptive_chunk, 
                    return_classification=True
                )
                
                # 손실 계산
                losses = self._compute_losses(outputs, batch)
                
                # 실제 역전파에 사용될 손실값 계산
                total_loss_for_backprop = losses['total_loss'] / self.accumulation_steps

            except Exception as e:
                print(f"Error in step {step}: {e}")
                self.optimizer.zero_grad()
                continue

            # Backward 처리
            try:
                if torch.isnan(total_loss_for_backprop) or torch.isinf(total_loss_for_backprop) or not total_loss_for_backprop.requires_grad:
                    continue

                total_loss_for_backprop.backward()

                # Gradient accumulation
                if step % self.accumulation_steps == 0:
                    max_grad_norm = self.config.get('max_grad_norm', 0.1)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                    
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()

            except Exception as e:
                print(f"Error in backward pass: {e}")
                self.optimizer.zero_grad()
                continue

            # 수정된 손실 로깅 - 실제 학습에 사용된 값들을 로깅
            # 총 손실은 accumulation으로 나눈 값
            total_losses['total'] += losses['total_loss'].item()

            for key in ['anc_dba', 'anc_nmse', 'anc_total', 'final_quality', 
                      's1_separation', 's2_separation', 'antinoise_constraint', 'classification']:
                loss_key = key + '_loss' if key != 'total' else 'total_loss'
                if key == 'anc_total':
                    loss_key = 'anc_total_loss'
                elif key == 'classification':
                    loss_key = 'classification_loss'

                if loss_key in losses:
                    actual_loss = losses[loss_key].item()
                    total_losses[key] += actual_loss
            
            if 'classification_accuracy' in losses:
                classification_accuracies.append(losses['classification_accuracy'])

            avg_total = total_losses['total'] / step
            avg_anc = total_losses['anc_total'] / step
            avg_separation = (total_losses['s1_separation'] + total_losses['s2_separation']) / 2 / step
            avg_classification = total_losses['classification'] / step

            postfix = {
            'Total': f"{avg_total:.4f}",
            'ANC': f"{avg_anc:.2f}dB", 
            'Sep': f"{avg_separation:.4f}",
            'Cls': f"{avg_classification:.4f}"
            }

            # 분류 정확도 (가능한 경우)
            if classification_accuracies:
                recent_acc = np.mean(classification_accuracies[-20:])  # 최근 20개 평균
                postfix['Acc'] = f"{recent_acc:.2f}"
            
            # 메모리 (간단히)
            postfix['Mem'] = f"{allocated:.1f}G"

            train_bar.set_postfix(postfix)

            # 상세 정보는 주기적으로 별도 출력 (매 50스텝)
            if step % 50 == 0:
                # 상세 분석
                avg_s1 = total_losses['s1_separation'] / step
                avg_s2 = total_losses['s2_separation'] / step
                avg_final_quality = total_losses['final_quality'] / step
                avg_anc_dba = total_losses['anc_dba'] / step
                avg_anc_nmse = total_losses['anc_nmse'] / step
                
                print(f"\nStep {step}/{num_batches} Detailed Metrics:")
                print(f"Separation: S1={avg_s1:.4f}, S2={avg_s2:.4f}")
                print(f"ANC: dBA={avg_anc_dba:.2f}dB, NMSE={avg_anc_nmse:.2f}dB")
                print(f"Final Quality: {avg_final_quality:.4f}")
                
                if classification_accuracies:
                    recent_acc = np.mean(classification_accuracies[-50:])
                    print(f"Classification: Loss={avg_classification:.4f}, Acc={recent_acc:.1%}")
                
                # 학습 상태 요약
                method_info = []
                if self.use_pit_loss:
                    method_info.append("PIT")
                if self.use_online_augment and aug_stats:
                    avg_aug = np.mean(aug_stats[-50:])
                    method_info.append(f"Aug={avg_aug:.2f}")
                
                if method_info:
                    print(f"Methods: {', '.join(method_info)}")
                print()  # 빈 줄

            # 정기적인 메모리 정리
            cleanup_interval = self.config.get('memory_cleanup_interval', 5)
            if step % cleanup_interval == 0:
                self.memory_manager.cleanup_memory(aggressive=True)

        # 평균 손실 계산
        avg_losses = {k: v/max(step, 1) for k, v in total_losses.items()}
        avg_losses['separation_loss'] = (avg_losses['s1_separation'] + avg_losses['s2_separation']) / 2
        
        if classification_accuracies:
            avg_losses['classification_accuracy'] = np.mean(classification_accuracies)

        return avg_losses
        
    def _validate(self):
        self.model.eval()

        total_losses = {
            'total': 0, 'anc_dba': 0, 'anc_nmse': 0, 'anc_total': 0,
            'final_quality': 0, 's1_separation': 0, 's2_separation': 0,
            'antinoise_constraint': 0, 'classification': 0
        }

        additional_metrics = {
            'processing_time': [],
            'classification_accuracy': []
        }

        num_batches = len(self.val_loader)

        val_bar = tqdm(self.val_loader, desc='Validation', ncols=100)

        for batch_idx, batch in enumerate(val_bar):
            try:
                start_time = time.time()

                # 데이터 전처리
                with torch.no_grad():
                    if 'sources' in batch:
                        # Training 타입 (실제로는 validation에서 안 나타남)
                        sources = batch['sources'].to(self.device)
                        mixed_input = sources.sum(dim=1, keepdim=True)
                        
                        # 정규화
                        m = mixed_input.mean(dim=-1, keepdim=True)
                        s = mixed_input.std(dim=-1, keepdim=True)
                        mixed_input = (mixed_input - m) / (s + 1e-9)
                    else:
                        # Validation 타입 (premixed)
                        mixed_input = batch['input'].to(self.device)

                    adaptive_chunk = self.memory_manager.get_safe_chunk_size()

                    # 모델 forward
                    outputs = self.model.forward_for_training(
                        mixed_input, 
                        chunk_len=adaptive_chunk, 
                        return_classification=True
                    )

                # 손실 계산을 위해 필요한 텐서들을 gradient 활성화
                key_tensors = ['s1_clean', 's2_noise', 's2_enhanced', 'enhanced_verification']
                for key in key_tensors:
                    if key in outputs and outputs[key] is not None:
                        outputs[key] = outputs[key].detach().requires_grad_(True)
                
                # 손실 계산
                losses = self._compute_losses(outputs, batch)

                # 기본 손실들 누적
                for key in total_losses.keys():
                    loss_key = key + '_loss' if key != 'total' else 'total_loss'
                    if key == 'anc_total':
                        loss_key = 'anc_total_loss'
                    elif key == 'classification':
                        loss_key = 'classification_loss'

                    if loss_key in losses:
                        total_losses[key] += losses[loss_key].item()
                
                # 분류 정확도 수집
                if 'classification_accuracy' in losses:
                    additional_metrics['classification_accuracy'].append(losses['classification_accuracy'])

                # 처리 시간 측정
                processing_time = time.time() - start_time
                additional_metrics['processing_time'].append(processing_time)

                avg_total = total_losses['total'] / (batch_idx + 1)
                avg_anc = total_losses['anc_total'] / (batch_idx + 1)
                
                postfix = {
                    'Loss': f"{avg_total:.4f}",
                    'ANC': f"{avg_anc:.2f}dB"
                }
                
                if additional_metrics['classification_accuracy']:
                    recent_acc = np.mean(additional_metrics['classification_accuracy'])
                    postfix['Acc'] = f"{recent_acc:.2f}"
                
                val_bar.set_postfix(postfix)

            except Exception as e:
                print(f"Validation error: {e}")
                continue

        # 평균 계산
        avg_losses = {k: v/max(num_batches, 1) for k, v in total_losses.items()}
        avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in additional_metrics.items()}

        avg_losses['separation_loss'] = (avg_losses['s1_separation'] + avg_losses['s2_separation']) / 2

        # 결합하여 반환
        result = {**avg_losses, **avg_metrics}

        return result
    
    def _log_epoch_summary(self, epoch, train_metrics, val_metrics):
        """에포크 완료 후 요약 출력"""
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*60}")
        
        # 🎯 핵심 지표 비교
        print(f"{'Metric':<20} {'Train':<12} {'Validation':<12} {'Status'}")
        print(f"{'-'*60}")
        
        # Total Loss
        train_total = train_metrics['total']
        val_total = val_metrics['total']
        status = "✅" if val_total < train_total * 1.2 else "⚠️"
        print(f"{'Total Loss':<20} {train_total:<12.4f} {val_total:<12.4f} {status}")
        
        # ANC
        train_anc = train_metrics['anc_total']
        val_anc = val_metrics['anc_total']
        status = "✅" if val_anc < train_anc + 1.0 else "⚠️"
        print(f"{'ANC (dB)':<20} {train_anc:<12.2f} {val_anc:<12.2f} {status}")
        
        # Separation
        train_sep = train_metrics.get('separation_loss', 0)
        val_sep = val_metrics.get('separation_loss', 0)
        status = "✅" if val_sep < train_sep * 1.3 else "⚠️"
        print(f"{'Separation':<20} {train_sep:<12.4f} {val_sep:<12.4f} {status}")
        
        # Classification
        if 'classification_accuracy' in val_metrics:
            train_acc = train_metrics.get('classification_accuracy', 0)
            val_acc = val_metrics['classification_accuracy']
            status = "✅" if val_acc > 0.6 else "⚠️"
            print(f"{'Classification':<20} {train_acc:<12.1%} {val_acc:<12.1%} {status}")
        
        print(f"{'='*60}\n")

    def calculate_composite_score(self, val_metrics):
        
        # 각 메트릭을 0-100 점수로 정규화
        def normalize_metric(value, target, direction='lower'):
            """
            direction: 'lower' = 낮을수록 좋음, 'higher' = 높을수록 좋음
            target: 목표값 (100점 기준)
            """
            if direction == 'lower':
                # 목표값보다 낮으면 100점, 높을수록 점수 감소
                if value <= target:
                    return 100.0
                else:
                    return max(0, 100 - (value - target) * 20)  # 초과시 빠르게 감점
            else:
                # 목표값보다 높으면 100점, 낮을수록 점수 감소  
                if value >= target:
                    return 100.0
                else:
                    return max(0, value / target * 100)
        
        # 각 성능 지표별 점수 계산
        anc_score = normalize_metric(
            val_metrics['anc_total'],  # abs() 제거
            target=-12.0,  # 음수 목표값
            direction='lower'  # 더 낮은(더 음수) 값이 좋음
        )
        
        separation_score = normalize_metric(
            val_metrics.get('separation_loss', val_metrics.get('s1_separation', 999)), 
            target=1.5,   # SI-SDR 1.5 목표
            direction='lower'
        )
        
        classification_score = normalize_metric(
            val_metrics.get('classification_accuracy', 0) * 100,
            target=80.0,  # 80% 정확도 목표
            direction='higher'
        )
        
        final_quality_score = normalize_metric(
            val_metrics['final_quality'],
            target=1.0,   # 최종 품질 손실 1.0 목표
            direction='lower'
        )
        
        # 가중 평균으로 종합 점수 계산
        weights = {
            'anc': 0.35,           # ANC 성능 35%
            'separation': 0.30,    # 분리 성능 30%
            'classification': 0.25, # 분류 성능 25%
            'final_quality': 0.10  # 최종 품질 10%
        }
        
        composite_score = (
            anc_score * weights['anc'] +
            separation_score * weights['separation'] + 
            classification_score * weights['classification'] +
            final_quality_score * weights['final_quality']
        )
        
        return {
            'composite_score': composite_score,
            'anc_score': anc_score,
            'separation_score': separation_score, 
            'classification_score': classification_score,
            'final_quality_score': final_quality_score
        }
        
    def _save_checkpoint(self, epoch, metrics, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'memory_stats': self.memory_manager.get_memory_usage(),
            'use_online_augment': self.use_online_augment,
            'use_pit_loss': self.use_pit_loss
        }

        # 항상 최신 모델 저장
        latest_path = os.path.join(self.checkpoint_path, 'latest_checkpoint.pth')
        torch.save(state, latest_path)

        # 종합 베스트 모델
        scores = self.calculate_composite_score(metrics)
        if scores['composite_score'] > getattr(self, 'best_composite_score', 0):
            self.best_composite_score = scores['composite_score']
            best_composite_path = os.path.join(self.checkpoint_path, 'best_composite.pth')
            torch.save(state, best_composite_path)
            print(f"NEW BEST COMPOSITE MODEL! Score: {scores['composite_score']:.1f}")
        
        # 개별 성능 베스트 모델들
        
        # ANC 베스트
        if metrics['anc_total'] < getattr(self, 'best_anc_performance', 0):  # 더 음수가 좋음
            self.best_anc_performance = metrics['anc_total']
            best_anc_path = os.path.join(self.checkpoint_path, 'best_anc.pth')
            torch.save(state, best_anc_path)
            print(f"NEW BEST ANC MODEL! ANC: {metrics['anc_total']:.2f}dB")
        
        # 분리 성능 베스트
        sep_loss = metrics.get('separation_loss', metrics.get('s1_separation', 999))
        if sep_loss < getattr(self, 'best_separation_loss', float('inf')):
            self.best_separation_loss = sep_loss
            best_sep_path = os.path.join(self.checkpoint_path, 'best_separation.pth')
            torch.save(state, best_sep_path)
            print(f"NEW BEST SEPARATION MODEL! Sep: {sep_loss:.3f}")
        
        # 분류 성능 베스트
        cls_acc = metrics.get('classification_accuracy', 0)
        if cls_acc > getattr(self, 'best_classification_accuracy', 0):
            self.best_classification_accuracy = cls_acc
            best_cls_path = os.path.join(self.checkpoint_path, 'best_classification.pth')
            torch.save(state, best_cls_path)
            print(f"NEW BEST CLASSIFICATION MODEL! Acc: {cls_acc:.1%}")

    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """메트릭 로깅"""
        if hasattr(self, 'writer'):
            # 기본 손실들
            for loss_name in ['total', 'anc_total', 'final_quality', 'classification']:
                self.writer.add_scalars(f'Loss/{loss_name}', {
                    'Train': train_metrics[loss_name],
                    'Val': val_metrics[loss_name]
                }, epoch)

            # PIT 사용 여부 로깅
            if self.use_pit_loss:
                self.writer.add_scalar('Training/PIT_Loss_Active', 1, epoch)
            else:
                self.writer.add_scalar('Training/PIT_Loss_Active', 0, epoch)

            # 추가 메트릭들
            if 'classification_accuracy' in val_metrics:
                self.writer.add_scalar('Metrics/Classification_Accuracy',
                                    val_metrics['classification_accuracy'], epoch)

            if 'processing_time' in val_metrics:
                self.writer.add_scalar('Metrics/Processing_Time',
                                     val_metrics['processing_time'], epoch)

            # 메모리 사용량
            allocated, reserved = self.memory_manager.get_memory_usage()
            self.writer.add_scalar('System/Memory_Allocated', allocated, epoch)
            self.writer.add_scalar('System/Memory_Reserved', reserved, epoch)

    def train(self):
        epochs = self.config.get('epochs', 15)
    
        # 베스트 성능 초기화
        self.best_composite_score = 0
        self.best_anc_performance = 0
        self.best_separation_loss = float('inf')
        self.best_classification_accuracy = 0

        print(f"Starting Mixed Training with Multi-Criteria Best Model Selection")
        print(f"Epochs: {epochs}")
        print(f"Training: Dynamic Mixing (spk1 + spk2)")
        print(f"Validation: Pre-mixed (mixtures, spk1, spk2)")
        print(f"PIT Loss: {self.use_pit_loss}")
        print(f"Online Augmentation: {self.use_online_augment}")
        
        # 🔧 Early Stopping 설정 정보 출력 (한 번만)
        print(f"Early Stopping Configuration:")
        print(f"Method: Multi-Metric (ANC: {self.early_stopping.metric_weights['anc_total']:.1%}, "
            f"Sep: {self.early_stopping.metric_weights['separation_loss']:.1%}, "
            f"Cls: {self.early_stopping.metric_weights['classification_accuracy']:.1%}, "
            f"Qual: {self.early_stopping.metric_weights['final_quality']:.1%})")
        print(f"   Patience: {self.early_stopping.patience}, Min Delta: {self.early_stopping.min_delta}")
        
        for epoch in range(1, epochs + 1):
            print(f"\n EPOCH {epoch}/{epochs}")

            try:
                # 훈련
                train_metrics = self._train_epoch(epoch)

                # 검증
                if self.val_loader:
                    val_metrics = self._validate()
                else:
                    val_metrics = train_metrics.copy()

                # 에포크 요약 출력
                self._log_epoch_summary(epoch, train_metrics, val_metrics)
                
                # 종합 점수 계산 (트레이너의 기존 함수)
                scores = self.calculate_composite_score(val_metrics)
                
                # Early Stopping 업데이트 (새로운 다중 메트릭 방식)
                self.early_stopping.update(val_metrics)

                # 스케줄러 업데이트
                composite_score_for_scheduler = (
                    val_metrics['anc_total'] * 0.4 +           # 낮을수록 좋음 (음수값)
                    val_metrics['separation_loss'] * 0.3 +     # 낮을수록 좋음
                    -val_metrics.get('classification_accuracy', 0) * 0.2 + # 높을수록 좋음 (음수 변환)
                    val_metrics['final_quality'] * 0.1         # 낮을수록 좋음
                )
                self.scheduler.step(composite_score_for_scheduler)
                
                # 종합 점수 표시 (트레이너의 기존 점수 + Early Stopping 점수)
                print(f"Trainer Composite Score: {scores['composite_score']:.1f} "
                    f"(ANC:{scores['anc_score']:.0f}, Sep:{scores['separation_score']:.0f}, "
                    f"Cls:{scores['classification_score']:.0f}, Qual:{scores['final_quality_score']:.0f})")

                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"LR: {current_lr:.2e}")

                # TensorBoard 로깅
                self._log_metrics(epoch, train_metrics, val_metrics)

                # 다중 베스트 모델 저장 (트레이너의 기존 함수)
                self._save_checkpoint(epoch, val_metrics)

                # 개선 상황 요약 (2 에포크마다, Early Stopping에서)
                if epoch % 2 == 0:
                    improvement_summary = self.early_stopping.get_improvement_summary()
                    print(f"Improvement Summary: {improvement_summary}")

                # 조기 종료 체크
                if self.early_stopping.should_stop():
                    best_scores = self.early_stopping.get_best_scores()
                    print(f"\n Early stopping after {epoch} epochs")
                    print(f"Early Stopping Best Composite Score: {best_scores['composite_score']:.4f}")
                    print(f"Best Individual Scores: {best_scores['individual_bests']}")
                    break

            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                self.memory_manager.cleanup_memory(aggressive=True)
                continue

            # 에포크 후 메모리 정리
            self.memory_manager.cleanup_memory(aggressive=True)

        print(f"\n MIXED TRAINING COMPLETED!")
        print(f" Best Composite Score: {self.best_composite_score:.1f}")
        print(f" Best ANC: {self.best_anc_performance:.2f}dB")
        print(f" Best Separation: {self.best_separation_loss:.3f}")
        print(f" Best Classification: {self.best_classification_accuracy:.1%}")
        print(f" Results saved in: {self.exp_path}")
        print(f"\n Available models:")
        print(f"   - best_composite.pth (종합 최고)")
        print(f"   - best_anc.pth (ANC 최고)")  
        print(f"   - best_separation.pth (분리 최고)")
        print(f"   - best_classification.pth (분류 최고)")