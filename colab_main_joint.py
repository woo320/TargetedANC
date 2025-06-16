import os
from trainers.sudormrf_trainer import ImprovedJointTrainerWithSudoRMRFMix
from project_utils.path_utils import auto_detect_data_type
from config.main_config import *

def main():
    # 1. 데이터 준비
    data_type = auto_detect_data_type(DEFAULT_DATASET_ROOT)

    if data_type == "ready":
        print("모든 데이터 준비 완료!")
    elif data_type == "training_only":
        print("Training 데이터만 있음 - Validation 없이 진행")
    else:
        print("데이터가 충분하지 않습니다!")
        return False

    config = main_config()

    # 3. 설정 요약 출력
    print(f"데이터: Training(동적믹싱) + Validation(premixed)")
    print(f"PIT Loss: {config['use_pit_loss']}")
    print(f"Classifier: {config['use_broadcast_classifier']}")
    print(f"배치: {config['batch_size']} × {config['accumulation_steps']} = {config['batch_size'] * config['accumulation_steps']}")
    print(f"에포크: {config['epochs']} (patience={config['early_stopping_patience']})")
    print(f"학습률: sep={config['separation_lr']:.0e}, noise={config['noise_lr']:.0e}, cls={config['classifier_lr']:.0e}")
    print(f"오디오: {config['max_audio_duration']}초, SR={config['sample_rate']}")
    print(f"메모리: {config['max_memory_gb']}GB 한계")
    print(f"손실 가중치: quality={config['loss_weights']['final_quality']}, anc={config['loss_weights']['anc_total']}")
    print(f"조기 종료: 다중 메트릭 (ANC:{config['early_stopping_weights']['anc_total']:.1%}, "
          f"Sep:{config['early_stopping_weights']['separation_loss']:.1%}, "
          f"Cls:{config['early_stopping_weights']['classification_accuracy']:.1%})")
    print(f"데이터 구조:")
    print(f"train/spk1/, train/spk2/ (동적믹싱)")
    print(f"val/mixtures/, val/spk1/, val/spk2/ (premixed)")

    # 4. 학습 실행
    try:
        trainer = ImprovedJointTrainerWithSudoRMRFMix(config)
        trainer.train()

        print(f"학습 완료!")
        print(f"결과: {trainer.exp_path}")
        return True

    except Exception as e:
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return False

def quick_test():
    """빠른 테스트용 설정"""
    print("빠른 테스트 모드")
    print("'epochs': 3")
    print("'limit_train_samples': 20")
    print("'limit_val_samples': 5")
    print("'max_audio_duration': 1.0")
    print("'early_stopping_patience': 3")
    print("'batch_size': 1")
    print("'accumulation_steps': 4")

if __name__ == "__main__":
    main()
