"""
🔧 수정된 메인 파라미터 관리 - Training(동적) + Validation(premix) 전용
"""
import os
import sys

# 🔧 프로젝트 경로 설정
sys.path.insert(0, "/content/drive/MyDrive/joint/code4")

from trainers.sudormrf_trainer import ImprovedJointTrainerWithSudoRMRFMix
from project_utils.path_utils import auto_detect_data_type
from config.model_paths import *

def main():
    """🚀 정리된 통합 파라미터 관리 - SudoRM-RF 학습"""

    print("🚀 SudoRM-RF 정리된 통합 파라미터 학습")
    print("💡 Training(동적믹싱) + Validation(premixed) 전용")
    print("="*60)

    # 1. 데이터 자동 감지
    print("🔍 데이터 확인 중...")
    data_type = auto_detect_data_type(DEFAULT_DATASET_ROOT)

    if data_type == "ready":
        print("✅ 모든 데이터 준비 완료!")
    elif data_type == "training_only":
        print("⚠️ Training 데이터만 있음 - Validation 없이 진행")
    else:
        print("❌ 데이터가 충분하지 않습니다!")
        print("💡 setup_mixed_data_directories()를 실행하여 구조를 생성하세요")
        return False

    # 2. 🔧 정리된 통합 파라미터 설정
    config = {
        # ===== 필수 경로들 =====
        'dataset_root': DEFAULT_DATASET_ROOT,
        'sudormrf_checkpoint': DEFAULT_SUDORMRF_CHECKPOINT,
        'wavenet_checkpoint': DEFAULT_WAVENET_CHECKPOINT,
        'wavenet_config': DEFAULT_WAVENET_CONFIG,
        'broadcast_classifier_checkpoint': DEFAULT_BROADCAST_CLASSIFIER_CHECKPOINT,

        # ===== 오디오 및 모델 기본 설정 =====
        'sample_rate': 16000,               # SR
        'eta_init_value': 0.1,              # ETA 초기값
        'max_audio_duration': 2.0,          # 초 단위 (메모리 절약)
        'use_broadcast_classifier': True,
        'use_pit_loss': True,               # SudoRM-RF PIT 손실 사용
        'use_online_augment': True,         # 온라인 증강 (Training만)

        # ===== 학습 설정 =====
        'epochs': 20,
        'batch_size': 4,                    # 메모리 안정성 우선
        'accumulation_steps': 16,           # 그라디언트 누적 (효과적 배치 = 16)

        # ===== 학습률 설정 =====
        'separation_lr': 3.5e-5,              # SudoRM-RF 분리 모델
        'noise_lr': 2.5e-5,                   # WaveNet 노이즈 모델
        'classifier_lr': 1.2e-4,              # BroadcastClassifier
        'eta_lr': 1.5e-5,                     # ETA 파라미터

        # ===== 🔧 조기 종료 설정 (다중 메트릭 방식) =====
        'early_stopping_patience': 7,       # 조기 종료용
        'min_delta': 0.0005,                # 조기 종료 최소 개선폭
        
        # ✅ 🆕 추가: 다중 메트릭 가중치 (EarlyStoppingManager용)
        'early_stopping_weights': {
            'anc_total': 0.25,               # ANC 성능 40%
            'separation_loss': 0.25,         # 분리 성능 30%
            'classification_accuracy': 0.25, # 분류 성능 20%
            'final_quality': 0.25            # 최종 품질 10%
        },
        
        # ===== 스케줄러 설정 =====
        'scheduler_patience': 2,             # LR 스케줄러용 (다른 목적)
        'scheduler_factor': 0.7,             # LR 감소 비율
        'min_lr': 1e-7,                     # 최소 학습률

        # ===== 메모리 관리 =====
        'max_memory_gb': 22.5,
        'memory_warning_threshold': 20.0,   # max_memory_gb의 89%
        'memory_critical_threshold': 21.5,  # max_memory_gb의 96%
        'memory_cleanup_interval': 5,       # 메모리 정리 주기 (스텝)

        # ===== 손실 가중치 =====
        'loss_weights': {
            'final_quality': 0.12,          # 최종 품질
            'anc_total': 0.28,              # ANC 총 손실
            'separation': 0.42,             # 분리 손실
            'classification': 0.15,         # 분류 손실
            'antinoise_constraint': 0.0
        },

        # ===== ANC 손실 세부 가중치 =====
        'anc_loss_weights': {
            'dba_weight': 0.5,              # dBA 손실 가중치
            'nmse_weight': 0.5,             # NMSE 손실 가중치
        },

        # ===== 적응적 청크 크기 설정 =====
        'adaptive_chunk_sizes': {
            'low_memory': 8000,             # 0.5초 (극도로 메모리 부족시)
            'medium_memory': 12000,         # 0.75초
            'normal_memory': 16000,         # 1초 (기본)
            'high_memory': 24000            # 1.5초 (최대)
        },

        # ===== 검증 및 저장 설정 =====
        'validation_interval': 1,           # 검증 주기 (에포크)
        'checkpoint_save_interval': 2,      # 체크포인트 저장 주기
        'save_audio_samples': True,
        'num_audio_samples_to_save': 3,

        # ===== 로깅 설정 =====
        'use_tensorboard': True,
        'log_interval_steps': 50,           # 훈련 로그 주기
        'tensorboard_log_interval': 20,     # TensorBoard 로그 주기
        'memory_debug_interval': 50,        # 메모리 체크 주기
        'progress_bar_refresh_rate': 10,    # Progress bar 업데이트 주기

        # ===== 그라디언트 및 최적화 =====
        'max_grad_norm': 0.05,              # 그라디언트 클리핑
        'use_gradient_checkpointing': True, # 그라디언트 체크포인팅

        # ===== 데이터 로딩 설정 =====
        'limit_train_samples': None,        # None = 전체 사용
        'limit_val_samples': None,
        'limit_test_samples': None,
        'dataloader_num_workers': 0,        # 멀티프로세싱 워커 수
        'dataloader_pin_memory': False,     # GPU 메모리 고정
        'dataloader_drop_last': True,       # 마지막 배치 드롭

        # ===== 고급 설정 =====
        'sef_clamp_range': (-10.0, 10.0),   # SEF 비선형성 클램핑 범위
        'fir_filter_safety_check': True,    # FIR 필터 안전성 체크
        'audio_eps': 1e-9,                  # 오디오 계산시 epsilon
        'target_db_for_save': -20,          # 저장시 정규화 레벨
        'target_antinoise_magnitude': 0.1,  # 안티노이즈 목표 크기

        # ===== 분류기 설정 =====
        'classification_window_len': 16000, # BroadcastClassifier 입력 길이
        'classification_threshold': 0.5,    # 이진 분류 임계값 (sigmoid > 0.5 = 방송)

        # ===== 디버깅 설정 =====
        'debug_loss_print_prob': 0.05,     # 손실 디버그 출력 확률 (5%)
        'debug_nan_check': True,            # NaN 체크 활성화
        'verbose_forward': False,           # 모델 forward 상세 로그 비활성화
        'verbose_collate': False,           # Collate 상세 로그 비활성화

        # ===== 실험 설정 =====
        'experiment_name': None,            # 실험 이름 (None = 자동 생성)
        'experiment_tags': ['sudormrf', 'pit_loss', 'broadcast_classifier', 'mixed_training'],
        'save_best_only': False,            # 최고 성능 모델만 저장

        # ===== 학습 안정화 =====
        'warmup_epochs': 2,                 # 처음 2 에포크는 낮은 학습률
        'warmup_lr_factor': 0.1,            # 워밍업 시 학습률 x0.1
        'classification_loss_warmup': True, # 분류 손실 점진적 증가
        'classification_target_accuracy': 0.8, # 목표 정확도 80% (현실적 목표)
    }

    # 3. 📊 설정 요약 출력
    print(f"🔧 정리된 설정 요약:")
    print(f"   📁 데이터: Training(동적믹싱) + Validation(premixed)")
    print(f"   🎯 PIT Loss: {'✅' if config['use_pit_loss'] else '❌'}")
    print(f"   🤖 Classifier: {'✅' if config['use_broadcast_classifier'] else '❌'}")
    print(f"   📦 배치: {config['batch_size']} × {config['accumulation_steps']} = {config['batch_size'] * config['accumulation_steps']}")
    print(f"   📈 에포크: {config['epochs']} (patience={config['early_stopping_patience']})")
    print(f"   🎛️ 학습률: sep={config['separation_lr']:.0e}, noise={config['noise_lr']:.0e}, cls={config['classifier_lr']:.0e}")
    print(f"   🎵 오디오: {config['max_audio_duration']}초, SR={config['sample_rate']}")
    print(f"   💾 메모리: {config['max_memory_gb']}GB 한계")
    print(f"   ⚖️ 손실 가중치: quality={config['loss_weights']['final_quality']}, anc={config['loss_weights']['anc_total']}")
    print(f"   🔧 조기 종료: 다중 메트릭 (ANC:{config['early_stopping_weights']['anc_total']:.1%}, "
          f"Sep:{config['early_stopping_weights']['separation_loss']:.1%}, "
          f"Cls:{config['early_stopping_weights']['classification_accuracy']:.1%})")
    print(f"   📊 데이터 구조:")
    print(f"      🎵 train/spk1/, train/spk2/ (동적믹싱)")
    print(f"      📊 val/mixtures/, val/spk1/, val/spk2/ (premixed)")

    # 4. 🚀 학습 실행
    try:
        trainer = ImprovedJointTrainerWithSudoRMRFMix(config)
        trainer.train()

        print(f"\n🎉 학습 완료!")
        print(f"📁 결과: {trainer.exp_path}")
        return True

    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test():
    """⚡ 빠른 테스트용 설정"""
    print("⚡ 빠른 테스트 모드")
    print("💡 main() 함수에서 다음 값들을 오버라이드하세요:")
    print("   'epochs': 3")
    print("   'limit_train_samples': 20")
    print("   'limit_val_samples': 5")
    print("   'max_audio_duration': 1.0")
    print("   'early_stopping_patience': 3")
    print("   'batch_size': 1")
    print("   'accumulation_steps': 4")

if __name__ == "__main__":
    main()