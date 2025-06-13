import torch
from config.constants import DEFAULT_SR

def sudormrf_dynamic_mix_collate_fn(batch, config=None):
    # 16kHz 설정 및 15초 길이 사용
    if config is not None:
        SR = config.get('sample_rate', 16000)
        max_duration = config.get('max_audio_duration', 15.0)
    else:
        SR = DEFAULT_SR # 16,000Hz
        max_duration = 15.0
    
    # s1, s2 배치 길이 확인
    all_lengths = []
    for item in batch:
        sources_length = item['sources'].shape[1] # 오디오 길이
        s1_length = len(item['separation_targets']['s1']) # 음성 길이
        s2_length = len(item['separation_targets']['s2']) # 소음 길이
        all_lengths.extend([sources_length, s1_length, s2_length])
    
    max_length_in_batch = max(all_lengths) if all_lengths else int(max_duration * SR) # all_lengths 리스트에서 가장 긴 길이 설정
    max_allowed_length = int(max_duration * SR) # 240,000 샘플 상한
    # 자르는 기준 설정
    target_length = min(max_length_in_batch, max_allowed_length)

    batch_sources = []
    batch_s1 = []
    batch_s2 = []
    batch_s1_filenames = []
    batch_s2_filenames = []

    for i, item in enumerate(batch):
        try:
            # 샘플에서 source, s1, s2 길이 추출
            sources = item['sources']
            s1 = item['separation_targets']['s1']
            s2 = item['separation_targets']['s2']
            
            sources_len = sources.shape[1]
            s1_len = len(s1)
            s2_len = len(s2)
            
            # 가장 짧은 길이 확인
            min_current_length = min(sources_len, s1_len, s2_len)
            
            # 가장 짧은 길이 기준으로 자르기
            sources_trimmed = sources[:, :min_current_length]
            s1_trimmed = s1[:min_current_length]
            s2_trimmed = s2[:min_current_length]
            
            # target_length 기준 부족한 만큼 zero padding
            if min_current_length < target_length:
                pad_length = target_length - min_current_length
                sources_padded = torch.cat([sources_trimmed, torch.zeros(2, pad_length)], dim=1)
                s1_padded = torch.cat([s1_trimmed, torch.zeros(pad_length)])
                s2_padded = torch.cat([s2_trimmed, torch.zeros(pad_length)])
            # target_length 만큼 자르기
            else:
                sources_padded = sources_trimmed[:, :target_length]
                s1_padded = s1_trimmed[:target_length]
                s2_padded = s2_trimmed[:target_length]
            
            # 자른 후 shape 검정
            assert sources_padded.shape == (2, target_length)
            assert s1_padded.shape == (target_length,)
            assert s2_padded.shape == (target_length,)

            # 자른 후 각 리스트에 추가
            batch_sources.append(sources_padded)
            batch_s1.append(s1_padded)
            batch_s2.append(s2_padded)
            batch_s1_filenames.append(item['s1_filename'])
            batch_s2_filenames.append(item['s2_filename'])
            
        except Exception as e:
            # 더미 데이터로 대체
            batch_sources.append(torch.zeros(2, target_length))
            batch_s1.append(torch.zeros(target_length))
            batch_s2.append(torch.zeros(target_length))
            batch_s1_filenames.append(f"error_{i}.wav")
            batch_s2_filenames.append(f"error_{i}.wav")

    # 샘플이 없다면 에러 발생
    if not batch_sources:
        raise ValueError("No valid items in batch")
    
    # 모든 sources의 크기가 동일한지 확인
    expected_shape = batch_sources[0].shape
    for i, sources in enumerate(batch_sources):
        if sources.shape != expected_shape:
            # 강제로 크기 맞추기
            if sources.shape[1] < expected_shape[1]:
                pad_len = expected_shape[1] - sources.shape[1]
                batch_sources[i] = torch.cat([sources, torch.zeros(2, pad_len)], dim=1)
            else:
                batch_sources[i] = sources[:, :expected_shape[1]]

    # 배치 생성: [B, 2, T]
    batch_sources = torch.stack(batch_sources)

    # SudoRM-RF 방식: 배치 레벨에서 믹싱과 정규화
    mixed_input = batch_sources.sum(dim=1, keepdim=True)  # [B, 1, T]

    # 차원 검증
    if mixed_input.dim() != 3:
        raise ValueError(f"Expected 3D mixed_input, got {mixed_input.dim()}D: {mixed_input.shape}")

    # SudoRM-RF 방식 정규화
    m = mixed_input.mean(dim=-1, keepdim=True)
    s = mixed_input.std(dim=-1, keepdim=True)
    mixed_input_normalized = (mixed_input - m) / (s + 1e-9)

    return {
        'sources': batch_sources,
        'input': mixed_input_normalized,
        'mix_stats': {'mean': m, 'std': s},
        'separation_targets': {
            's1': torch.stack(batch_s1),
            's2': torch.stack(batch_s2)
        },
        's1_filenames': batch_s1_filenames,
        's2_filenames': batch_s2_filenames
    }

def improved_collate_fn(batch, config=None):
    """개선된 배치 콜레이트 (기존 방식, 정리 버전)"""
    # config에서 sample_rate 가져오기
    if config is not None:
        SR = config.get('sample_rate', 16000)
        max_duration = config.get('max_audio_duration', 15.0)
    else:
        from config.constants import DEFAULT_SR
        SR = DEFAULT_SR
        max_duration = 15.0
    
    # 모든 아이템의 길이 확인
    all_lengths = []
    for item in batch:
        input_length = len(item['input'])
        s1_length = len(item['separation_targets']['s1'])
        s2_length = len(item['separation_targets']['s2'])
        all_lengths.extend([input_length, s1_length, s2_length])
    
    # 최대 길이 결정
    max_length_in_batch = max(all_lengths) if all_lengths else int(max_duration * SR)
    max_allowed_length = int(max_duration * SR)
    target_length = min(max_length_in_batch, max_allowed_length)

    batch_input = []
    batch_s1 = []
    batch_s2 = []
    batch_filenames = []

    for i, item in enumerate(batch):
        try:
            # 각 아이템의 길이 통일
            input_audio = item['input']
            s1_audio = item['separation_targets']['s1']
            s2_audio = item['separation_targets']['s2']
            
            # 가장 짧은 길이로 맞추기
            min_current_length = min(len(input_audio), len(s1_audio), len(s2_audio))
            
            input_trimmed = input_audio[:min_current_length]
            s1_trimmed = s1_audio[:min_current_length]
            s2_trimmed = s2_audio[:min_current_length]
            
            # target_length에 맞춰 패딩 또는 자르기
            if min_current_length < target_length:
                pad_length = target_length - min_current_length
                input_padded = torch.cat([input_trimmed, torch.zeros(pad_length)])
                s1_padded = torch.cat([s1_trimmed, torch.zeros(pad_length)])
                s2_padded = torch.cat([s2_trimmed, torch.zeros(pad_length)])
            else:
                input_padded = input_trimmed[:target_length]
                s1_padded = s1_trimmed[:target_length]
                s2_padded = s2_trimmed[:target_length]

            batch_input.append(input_padded)
            batch_s1.append(s1_padded)
            batch_s2.append(s2_padded)
            batch_filenames.append(item['filename'])
            
        except Exception:
            # 더미 데이터로 대체
            batch_input.append(torch.zeros(target_length))
            batch_s1.append(torch.zeros(target_length))
            batch_s2.append(torch.zeros(target_length))
            batch_filenames.append(f"error_{i}.wav")

    return {
        'input': torch.stack(batch_input),
        'separation_targets': {
            's1': torch.stack(batch_s1),
            's2': torch.stack(batch_s2)
        },
        'filename': batch_filenames
    }