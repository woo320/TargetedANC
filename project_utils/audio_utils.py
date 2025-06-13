import torch

def standardize_audio_dims(audio):
    """오디오 차원 표준화"""
    if audio.dim() == 1:
        # [T] -> [1, 1, T]
        result = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        # [B, T] -> [B, 1, T]
        result = audio.unsqueeze(1)
    elif audio.dim() == 3:
        # [B, C, T] - 이미 올바른 형태
        result = audio
    elif audio.dim() == 4:
        # [B, C, 1, T] -> [B, C, T] (중간 차원 제거)
        result = audio.squeeze(2)
        # 혹시 [B, 1, C, T] 형태라면
        if result.shape[1] > result.shape[2] and result.shape[2] == 1:
            result = result.squeeze(2)
    else:
        raise ValueError(f"Unsupported audio dimensions: {audio.shape}")
    
    return result