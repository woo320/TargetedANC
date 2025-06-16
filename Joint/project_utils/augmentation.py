import torch

def online_augment_sudormrf(sources):
    """
    [역할]
    온라인 데이터 증강(C-SudoRM-RF++ 방식)
    실시간으로 배치 내에서 소스들을 섞고 스케일링하여 학습 데이터의 다양성 증가
        
    1. 배치 내에서 각 소스별로 랜덤 순열 생성 (조합 다양화)
    2. 각 배치, 소스별로 랜덤 스케일링 적용 (볼륨 다양화)
    """

    device = sources.device
    B, n_src, T = sources.shape

    # 1.
    perm = [sources[torch.randperm(B, device=device), i] for i in range(n_src)]
    wavs = torch.stack(perm, dim=1)

    # 2.
    scales = torch.rand(B, n_src, 1, device=device) * 0.4 + 0.8
    
    return wavs * scales