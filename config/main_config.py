from config.model_paths import *

def main_config():
    config = {
        # 데이터 / 모델 가중치 로드
        'dataset_root': DEFAULT_DATASET_ROOT,
        'sudormrf_checkpoint': DEFAULT_SUDORMRF_CHECKPOINT,
        'wavenet_checkpoint': DEFAULT_WAVENET_CHECKPOINT,
        'wavenet_config': DEFAULT_WAVENET_CONFIG,
        'broadcast_classifier_checkpoint': DEFAULT_BROADCAST_CLASSIFIER_CHECKPOINT,

        # 오디오 및 모델 기본 설정
        'sample_rate': 16000,               # SR
        'eta_init_value': 0.1,              # ETA 초기값
        'max_audio_duration': 2.0,          # 2초 단위
        'use_broadcast_classifier': True,
        'use_pit_loss': True,               # PIT 손실 함수
        'use_online_augment': True,         # 데이터 증강 (학습용)

        # 학습 설정
        'epochs': 20,
        'batch_size': 4,
        'accumulation_steps': 16,

        # 학습률 설정
        'separation_lr': 3.5e-5,              # (분리) SudoRM-RF 모델
        'noise_lr': 2.5e-5,                   # (저감) WaveNet-VNNs 모델
        'classifier_lr': 1.2e-4,              # (분리 판별) Audio Segment Classifier
        'eta_lr': 1.5e-5,                     # ETA 파라미터

        # EarlyStopping 설정
        'early_stopping_patience': 7,
        'min_delta': 0.0005,
        
        # 추가: 다중 메트릭 가중치 (EarlyStoppingManager용)
        'early_stopping_weights': {
            'anc_total': 0.25,               # ANC 성능 40%
            'separation_loss': 0.25,         # 분리 성능 30%
            'classification_accuracy': 0.25, # 분류 성능 20%
            'final_quality': 0.25            # 최종 품질 10%
        },
        
        # 스케줄러 설정
        'scheduler_patience': 2,             # LR 스케줄러용 (다른 목적)
        'scheduler_factor': 0.7,             # LR 감소 비율
        'min_lr': 1e-7,                      # 최소 학습률

        # 메모리 관리
        'max_memory_gb': 22.5,
        'memory_warning_threshold': 20.0,   # max_memory_gb의 89%
        'memory_critical_threshold': 21.5,  # max_memory_gb의 96%
        'memory_cleanup_interval': 5,       # 메모리 정리 주기 (스텝)

        # 손실 가중치
        'loss_weights': {
            'final_quality': 0.12,          # 최종 품질
            'anc_total': 0.28,              # ANC 총 손실
            'separation': 0.42,             # 분리 손실
            'classification': 0.15,         # 분류 손실
            'antinoise_constraint': 0.0
        },

        # ANC 손실 세부 가중치
        'anc_loss_weights': {
            'dba_weight': 0.5,              # dBA 손실 가중치
            'nmse_weight': 0.5,             # NMSE 손실 가중치
        },

        # 적응적 청크 크기 설정
        'adaptive_chunk_sizes': {
            'low_memory': 8000,             # 0.5초 (극도로 메모리 부족시)
            'medium_memory': 12000,         # 0.75초
            'normal_memory': 16000,         # 1초 (기본)
            'high_memory': 24000            # 1.5초 (최대)
        },

        # 검증 및 저장 설정
        'validation_interval': 1,           # 검증 주기 (에포크)
        'checkpoint_save_interval': 2,      # 체크포인트 저장 주기
        'save_audio_samples': True,
        'num_audio_samples_to_save': 3,

        # 로깅 설정
        'use_tensorboard': True,
        'log_interval_steps': 50,           # 훈련 로그 주기
        'tensorboard_log_interval': 20,     # TensorBoard 로그 주기
        'memory_debug_interval': 50,        # 메모리 체크 주기
        'progress_bar_refresh_rate': 10,    # Progress bar 업데이트 주기

        # 그라디언트 및 최적화
        'max_grad_norm': 0.05,              # 그라디언트 클리핑
        'use_gradient_checkpointing': True, # 그라디언트 체크포인팅

        # 데이터 로딩 설정
        'limit_train_samples': None,        # 전체 데이터 사용
        'limit_val_samples': None,
        'limit_test_samples': None,
        'dataloader_num_workers': 0,
        'dataloader_pin_memory': False,
        'dataloader_drop_last': True,

        # 정규화 설정
        'sef_clamp_range': (-10.0, 10.0),   # SEF 비선형성 클램핑
        'fir_filter_safety_check': True,    # FIR 필터 안전성 체크
        'audio_eps': 1e-9,                  # 오디오 계산시 epsilon
        'target_db_for_save': -20,          # 저장시 정규화 레벨
        'target_antinoise_magnitude': 0.1,  # 안티노이즈 목표 크기

        # 분류기 설정
        'classification_window_len': 16000, # BroadcastClassifier 입력 길이
        'classification_threshold': 0.5,    # 이진 분류 임계값 (sigmoid > 0.5 = 방송)

        # 디버깅 설정
        'debug_loss_print_prob': 0.05,     # 손실 디버그 출력 확률 (5%)
        'debug_nan_check': True,            # NaN 체크 활성화
        'verbose_forward': False,           # 모델 forward 상세 로그 비활성화
        'verbose_collate': False,           # Collate 상세 로그 비활성화

        # 실험 설정
        'experiment_name': None,            # 실험 이름 (None = 자동 생성)
        'experiment_tags': ['sudormrf', 'pit_loss', 'broadcast_classifier', 'mixed_training'],
        'save_best_only': False,            # 최고 성능 모델만 저장

        # 학습 안정화
        'warmup_epochs': 2,                 # 처음 2 에포크는 낮은 학습률
        'warmup_lr_factor': 0.1,            # 워밍업 시 학습률 x0.1
        'classification_loss_warmup': True, # 분류 손실 점진적 증가
        'classification_target_accuracy': 0.8, # 목표 정확도 80% (현실적 목표)
    }

    return config