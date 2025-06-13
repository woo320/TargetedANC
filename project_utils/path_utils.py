"""
Training용 동적믹스 + Validation용 premix 지원 경로 유틸리티
"""
import os

def setup_mixed_data_directories(base_dir="/content/drive/MyDrive/final_data"):
    """Training용 동적믹스 + Validation용 premix 디렉토리 구조 생성"""
    
    # Training: 동적 믹스용
    train_subdirs = ['spk1', 'spk2']
    
    # Validation/Test: Pre-mixed용  
    val_test_subdirs = ['mixtures', 'spk1', 'spk2']
    
    created_dirs = []

    # Training 디렉토리
    for subdir in train_subdirs:
        dir_path = os.path.join(base_dir, 'train', subdir)
        os.makedirs(dir_path, exist_ok=True)
        created_dirs.append(dir_path)
    
    # Validation/Test 디렉토리
    for split in ['val', 'test']:
        for subdir in val_test_subdirs:
            dir_path = os.path.join(base_dir, split, subdir)
            os.makedirs(dir_path, exist_ok=True)
            created_dirs.append(dir_path)

    print("📁 Created mixed directory structure:")
    print("📂 TRAINING (Dynamic Mix Only):")
    for subdir in train_subdirs:
        print(f"   ✅ /train/{subdir}/")
    
    print("📂 VALIDATION/TEST (Pre-mixed Only):")
    for split in ['val', 'test']:
        for subdir in val_test_subdirs:
            print(f"   ✅ /{split}/{subdir}/")

    print(f"\n💡 Usage:")
    print(f"   🎵 Training: Put spk1/spk2 files in /train/spk1/, /train/spk2/")
    print(f"   📊 Validation: Put premixed files in /val/mixtures/, /val/spk1/, etc.")

    return created_dirs

def check_training_data(base_dir="/content/drive/MyDrive/final_data"):
    """Training용 동적 믹스 데이터 확인"""
    print(f"🔍 Checking TRAINING data (dynamic mix) in: {base_dir}/train/")
    
    train_subdirs = ['spk1', 'spk2']
    total_files = 0
    has_training_data = False

    for subdir in train_subdirs:
        dir_path = os.path.join(base_dir, 'train', subdir)
        
        if os.path.exists(dir_path):
            wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            file_count = len(wav_files)
            total_files += file_count
            
            if file_count > 0:
                has_training_data = True
            
            status = "✅" if file_count > 0 else "⚠️"
            print(f"   {status} train/{subdir}: {file_count} files")
            
            if file_count > 0:
                sample_files = wav_files[:3]
                for i, filename in enumerate(sample_files):
                    print(f"      {i+1}. {filename}")
                if file_count > 3:
                    print(f"      ... and {file_count-3} more")
        else:
            print(f"   ❌ train/{subdir}: Directory not found")
    
    print(f"📊 Training files: {total_files}")
    return has_training_data

def check_validation_data(base_dir="/content/drive/MyDrive/final_data"):
    """Validation용 pre-mixed 데이터 확인"""
    print(f"🔍 Checking VALIDATION data (pre-mixed) in: {base_dir}/val/")
    
    val_subdirs = ['mixtures', 'spk1', 'spk2']
    total_files = 0
    has_validation_data = False

    for subdir in val_subdirs:
        dir_path = os.path.join(base_dir, 'val', subdir)
        
        if os.path.exists(dir_path):
            wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            file_count = len(wav_files)
            total_files += file_count
            
            if file_count > 0:
                has_validation_data = True
            
            status = "✅" if file_count > 0 else "⚠️"
            print(f"   {status} val/{subdir}: {file_count} files")
            
            if file_count > 0:
                sample_files = wav_files[:3]
                for i, filename in enumerate(sample_files):
                    print(f"      {i+1}. {filename}")
                if file_count > 3:
                    print(f"      ... and {file_count-3} more")
        else:
            print(f"   ❌ val/{subdir}: Directory not found")
    
    print(f"📊 Validation files: {total_files}")
    return has_validation_data

def auto_detect_data_type(base_dir="/content/drive/MyDrive/final_data"):
    """데이터 타입 자동 감지 - Training(동적) + Validation(premix) 전용"""
    print(f"🕵️ Auto-detecting data structure in: {base_dir}")
    print("="*60)

    # Training 데이터 확인 (동적믹스용)
    has_training = check_training_data(base_dir)
    print("\n" + "-"*40)
    
    # Validation 데이터 확인 (premix용)
    has_validation = check_validation_data(base_dir)
    
    print("\n" + "="*60)
    print(f"🎯 Detection Results:")
    print(f"   Training (spk1/spk2 for dynamic mix): {'✅' if has_training else '❌'}")
    print(f"   Validation (mixtures/spk1/spk2): {'✅' if has_validation else '❌'}")

    if has_training and has_validation:
        print(f"\n💡 Perfect! Ready for training:")
        print(f"   🎵 Training will use dynamic mixing (spk1 + spk2)")
        print(f"   📊 Validation will use pre-mixed data")
        return "ready"
    elif has_training:
        print(f"\n⚠️ Training data found, but no validation data")
        print(f"   💡 Training possible, but no validation available")
        return "training_only"
    elif has_validation:
        print(f"\n⚠️ Only validation data found") 
        print(f"   💡 Need training data (spk1/spk2 folders)")
        return "validation_only"
    else:
        print(f"\n❌ No suitable data found!")
        print(f"💡 Please run setup_mixed_data_directories() to create structure")
        return "none"