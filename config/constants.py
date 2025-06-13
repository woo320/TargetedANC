"""
수정된 constants.py - 메인 config에서 관리하도록 변경
"""
import torch

# 🔧 기본값들 (메인 config에서 오버라이드 가능)
DEFAULT_SR = 16000
DEFAULT_ETA = 0.1

# 메모리 관리 기본값 (메인 config에서 오버라이드 가능)
DEFAULT_MAX_MEMORY_GB = 22.5
DEFAULT_MEMORY_WARNING_THRESHOLD = 20.0
DEFAULT_MEMORY_CRITICAL_THRESHOLD = 21.5

# 하위호환성을 위한 별칭 (기존 코드가 작동하도록)
SR = DEFAULT_SR
ETA = DEFAULT_ETA
MAX_MEMORY_GB = DEFAULT_MAX_MEMORY_GB
MEMORY_WARNING_THRESHOLD = DEFAULT_MEMORY_WARNING_THRESHOLD
MEMORY_CRITICAL_THRESHOLD = DEFAULT_MEMORY_CRITICAL_THRESHOLD

# ANC 필터 경로 (이건 고정값이므로 유지)
PRI_PATH = "/content/drive/MyDrive/joint/WaveNet-VNNs-for-ANC/WaveNet_VNNs/pri_channel.mat"
SEC_PATH = "/content/drive/MyDrive/joint/WaveNet-VNNs-for-ANC/WaveNet_VNNs/sec_channel.mat"

# 필터 로드 (안전한 로딩)
try:
    from scipy.io import loadmat
    PRI = torch.tensor(loadmat(PRI_PATH)["pri_channel"].squeeze(), dtype=torch.float)
    SEC = torch.tensor(loadmat(SEC_PATH)["sec_channel"].squeeze(), dtype=torch.float)
except Exception as e:
    print(f"⚠️ Warning: Could not load ANC filters: {e}")
    # 더미 필터 생성
    PRI = torch.randn(64) * 0.01
    SEC = torch.randn(64) * 0.01

def get_sr_from_config(config):
    """config에서 sample_rate 가져오기 (하위호환성)"""
    return config.get('sample_rate', DEFAULT_SR)

def get_eta_from_config(config):
    """config에서 eta_init_value 가져오기 (하위호환성)"""
    return config.get('eta_init_value', DEFAULT_ETA)