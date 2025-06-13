"""
방송/소음 판별 분류기 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BroadcastClassifier(nn.Module):
    """
    방송(s1)/소음(s2) 판별 분류기
    입력: (batch, 1, window_len)
    출력: (batch, 1) logits (sigmoid 전 값)
    """
    def __init__(self, window_len=16000):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=31, stride=2, padding=15)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=31, stride=2, padding=15)
        self.bn3   = nn.BatchNorm1d(64)
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
      """강력한 차원 검증이 있는 forward"""
      original_shape = x.shape
      
      # 강제로 3차원으로 만들기
      while x.dim() > 3:
          x = x.squeeze()
      
      if x.dim() == 2:
          x = x.unsqueeze(1)
      
      if x.dim() != 3:
          raise ValueError(f"Cannot process input: original={original_shape}, processed={x.shape}")
      
      # Conv1D 요구사항: [batch, channels, length]
      batch_size, channels, length = x.shape
      
      x = F.relu(self.bn1(self.conv1(x)))
      x = F.relu(self.bn2(self.conv2(x)))
      x = F.relu(self.bn3(self.conv3(x)))
      x = self.global_avgpool(x).squeeze(2)  # (B, 64)
      return self.fc(x)  # (B, 1)

    def predict_proba(self, x):
        """확률값 반환 (0~1)"""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x, threshold=0.5):
        """이진 예측 (0 또는 1)"""
        proba = self.predict_proba(x)
        return (proba >= threshold).float()