import torch

def online_augment_sudormrf(sources):
    """SudoRM-RF 방식의 온라인 증강"""
    device = sources.device
    B, n_src, T = sources.shape

    # 배치 내에서 랜덤 순열 생성
    perm = [sources[torch.randperm(B, device=device), i] for i in range(n_src)]
    wavs = torch.stack(perm, dim=1)

    # 랜덤 스케일링 (0.5 ~ 1.5 범위)
    scales = torch.rand(B, n_src, 1, device=device) + 0.5
    return wavs * scales