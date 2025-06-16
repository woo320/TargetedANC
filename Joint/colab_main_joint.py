import sys
import importlib.util
import os

# 필요한 모듈들을 직접 로드
def load_module_directly(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    # 1. sudormrf_trainer 로드
    trainer_module = load_module_directly(
        "sudormrf_trainer", 
        os.path.join(os.getcwd(), "trainers", "sudormrf_trainer.py")
    )
    ImprovedJointTrainerWithSudoRMRFMix = trainer_module.ImprovedJointTrainerWithSudoRMRFMix
    print("✓ trainers.sudormrf_trainer 로드 성공!")
    
    # 2. path_utils 로드
    path_utils_module = load_module_directly(
        "path_utils",
        os.path.join(os.getcwd(), "project_utils", "path_utils.py")
    )
    auto_detect_data_type = path_utils_module.auto_detect_data_type
    print("✓ project_utils.path_utils 로드 성공!")
    
    # 3. main_config 로드
    config_module = load_module_directly(
        "main_config",
        os.path.join(os.getcwd(), "config", "main_config.py")
    )
    main_config = config_module.main_config
    DEFAULT_DATASET_ROOT = config_module.DEFAULT_DATASET_ROOT
    print("✓ config.main_config 로드 성공!")
    
    print("\n모든 모듈 로드 완료! 이제 main 함수 실행...")
    
    # 4. main 함수 실행
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
    
    # main 함수 실행
    main()
    
except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()
