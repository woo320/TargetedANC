import os

def setup_mixed_data_directories(base_dir="/content/drive/MyDrive/final_data"):
    """
    [역할]
    Mix 데이터 디렉토리 구조 생성
    - Training: 동적 믹싱용 (spk1, spk2)
    - Validation/Test: Pre-mixed용 (mixtures, spk1, spk2)
    """
    
    train_subdirs = ['spk1', 'spk2']
    
    val_test_subdirs = ['mixtures', 'spk1', 'spk2'] # mixtures : 사전 믹싱된 파일
    
    created_dirs = []

    for subdir in train_subdirs:
        dir_path = os.path.join(base_dir, 'train', subdir)
        os.makedirs(dir_path, exist_ok=True)
        created_dirs.append(dir_path)
    
    for split in ['val', 'test']:
        for subdir in val_test_subdirs:
            dir_path = os.path.join(base_dir, split, subdir)
            os.makedirs(dir_path, exist_ok=True)
            created_dirs.append(dir_path)

    print("혼합 디렉토리 구조 생성 완료:")
    print("TRAINING (동적 믹싱 전용):")
    for subdir in train_subdirs:
        print(f"   /train/{subdir}/")
    
    print("VALIDATION/TEST (Pre-mixed 전용):")
    for split in ['val', 'test']:
        for subdir in val_test_subdirs:
            print(f"   /{split}/{subdir}/")

    print(f"\n사용 방법:")
    print(f"   Training: spk1/spk2 파일을 /train/spk1/, /train/spk2/에 저장")
    print(f"   Validation: 믹싱된 파일을 /val/mixtures/, /val/spk1/ 등에 저장")

    return created_dirs

def check_training_data(base_dir="/content/drive/MyDrive/final_data"):
    """
    [역할] spk1, spk2에 파일들이 있는지 확인
    """

    print(f"TRAINING 데이터 확인 (동적 믹싱용): {base_dir}/train/")
    
    train_subdirs = ['spk1', 'spk2']
    total_files = 0
    has_training_data = False

    for subdir in train_subdirs:
        dir_path = os.path.join(base_dir, 'train', subdir)
        
        if os.path.exists(dir_path):
            wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            file_count = len(wav_files)
            total_files += file_count   # .wav 파일 개수 확인
            
            if file_count > 0:
                has_training_data = True
            
            status = "OK" if file_count > 0 else "FAIL"
            print(f"   {status} train/{subdir}: {file_count}개 파일")
            
            if file_count > 0:
                sample_files = wav_files[:3]
                for i, filename in enumerate(sample_files):
                    print(f"      {i+1}. {filename}")
                if file_count > 3:
                    print(f"      ... 외 {file_count-3}개 더")
        else:
            print(f"   train/{subdir}: 디렉토리 없음")
    
    print(f"Training 파일 총 개수: {total_files}")
    return has_training_data

def check_validation_data(base_dir="/content/drive/MyDrive/final_data"):
    """
    [역할] mixtures, spk1, spk2에 파일들이 있는지 확인
    """

    print(f"VALIDATION 데이터 확인 (pre-mixed용): {base_dir}/val/")
    
    val_subdirs = ['mixtures', 'spk1', 'spk2']
    total_files = 0
    has_validation_data = False

    for subdir in val_subdirs:
        dir_path = os.path.join(base_dir, 'val', subdir)
        
        if os.path.exists(dir_path):
            wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            file_count = len(wav_files)
            total_files += file_count   # .wav 파일 개수 확인
            
            if file_count > 0:
                has_validation_data = True
            
            status = "OK" if file_count > 0 else "FAIL"
            print(f"   {status} val/{subdir}: {file_count}개 파일")
            
            if file_count > 0:
                sample_files = wav_files[:3]
                for i, filename in enumerate(sample_files):
                    print(f"      {i+1}. {filename}")
                if file_count > 3:
                    print(f"      ... 외 {file_count-3}개 더")
        else:
            print(f"   val/{subdir}: 디렉토리 없음")
    
    print(f"Validation 파일 총 개수: {total_files}")
    return has_validation_data

def auto_detect_data_type(base_dir="/content/drive/MyDrive/final_data"):
    """
    [역할]
    데이터 타입 자동 감지
    전체 데이터 구조를 분석하여 학습 가능 여부를 판단
    """

    print(f"데이터 구조 자동 감지: {base_dir}")
    print("="*60)

    has_training = check_training_data(base_dir)
    print("\n" + "-"*40)
    
    has_validation = check_validation_data(base_dir)
    
    print("\n" + "="*60)
    print(f"감지 결과:")
    print(f"   Training (spk1/spk2 동적 믹싱용): {'OK' if has_training else 'FAIL'}")
    print(f"   Validation (mixtures/spk1/spk2): {'OK' if has_validation else 'FAIL'}")

    if has_training and has_validation:
        print(f"\n 완벽! 학습 준비 완료:")
        print(f"   Training은 동적 믹싱 사용 (spk1 + spk2)")
        print(f"   Validation은 pre-mixed 데이터 사용")
        return "ready"
    elif has_training:
        print(f"\nTraining 데이터만 발견, Validation 데이터 없음")
        print(f"   학습은 가능하지만 검증 불가")
        return "training_only"
    elif has_validation:
        print(f"\nValidation 데이터만 발견") 
        print(f"   Training 데이터 필요 (spk1/spk2 폴더)")
        return "validation_only"
    else:
        print(f"\n적절한 데이터를 찾을 수 없음!")
        print(f"setup_mixed_data_directories() 실행하여 구조 생성")
        return "none"