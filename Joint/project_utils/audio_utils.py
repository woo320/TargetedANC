import torch

def standardize_audio_dims(audio):
    """
    [역할] 오디오 텐서 차원 표준화
    """
    
    if audio.dim() == 1:
        # [T] -> [1, 1, T]
        result = audio.unsqueeze(0).unsqueeze(0)
        
    elif audio.dim() == 2:
        # [B, T] -> [B, 1, T]
        result = audio.unsqueeze(1)
        
    elif audio.dim() == 3:
        # [B, C, T]
        result = audio
        
    elif audio.dim() == 4:
        # [B, C, 1, T] -> [B, C, T]
        result = audio.squeeze(2)
        
        # [B, 1, C, T] -> 재배치
        if result.shape[1] > result.shape[2] and result.shape[2] == 1:
            result = result.squeeze(2)
            
    else:
        raise ValueError(f"지원하지 않는 오디오 차원: {audio.shape}")
    
    return result