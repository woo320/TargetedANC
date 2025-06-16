
#안내방송음/그 외 소음 판별 분류기 모델
import torch
import torch.nn as nn
import torch.nn.functional as F

class BroadcastClassifier(nn.Module):
    """
    1초 길이의 파형 입력으로 안내방송음과 그 외 소음을 구분하는 1D CNN 분류기.
    입력:  (batch, 1, window_len) float32
    출력: (batch, 1) - 각 구간별 방송/소음 로짓값 (sigmoid 전)
    """
    
    #window_len 안쓰이는것 같음.
    def __init__(self, window_len=16000):
        super().__init__()
        # 1차원 합성곱 계층 3개로 구성된 CNN 기반 분류기 구조
        self.conv1 = nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=31, stride=2, padding=15)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=31, stride=2, padding=15)
        self.bn3   = nn.BatchNorm1d(64)
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)  # 시간축 평균
        self.fc = nn.Linear(64, 1)  # 최종 이진분류용 logit 출력

    def forward(self, x):
        # 입력 차원을 3D(batch, channel, length)로 통일
        original_shape = x.shape
        # 과한 차원 제거
        while x.dim() > 3:
            x = x.squeeze()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Cannot process input: original={original_shape}, processed={x.shape}")
        
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_avgpool(x).squeeze(2)
        return self.fc(x)  # (batch, 1)

    def predict_proba(self, x):
        # sigmoid 적용 후 확률값 반환 (0: 그 외 소음, 1: 안내방송음)
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x, threshold=0.5):
        # threshold 이상이면 안내방송음(1), 아니면 그 외 소음(0)
        proba = self.predict_proba(x)
        return (proba >= threshold).float()
