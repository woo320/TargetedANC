"""
모델 체크포인트 경로 설정
"""
# 기본 모델 체크포인트 경로들
DEFAULT_SUDORMRF_CHECKPOINT = "/weight/separate.pt"
DEFAULT_WAVENET_CHECKPOINT = "/weight/reduction.pth"
DEFAULT_WAVENET_CONFIG = "/weight/config_opt_210.json"

# BroadcastClassifier 체크포인트
DEFAULT_BROADCAST_CLASSIFIER_CHECKPOINT = "/weight/classifier.pth"

# 데이터셋 기본 경로
DEFAULT_DATASET_ROOT = "/content/drive/MyDrive/final_data"
DEFAULT_DATA_ROOT = "/content/drive/MyDrive/joint/data"

# 결과 저장 기본 경로
DEFAULT_RESULT_ROOT = "/result"