import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from config.constants import SR
from config.constants import DEFAULT_SR

class ImprovedAudioDataset(Dataset):
    def __init__(self, dataset_root, split='train', max_samples=None, max_duration=15.0, config=None):
        self.dataset_root = dataset_root
        self.split = split
        self.max_duration = max_duration
        
        # config에서 sample_rate 가져오기
        if config is not None:
            self.SR = config.get('sample_rate', 16000)
        else:
            self.SR = DEFAULT_SR
        
        # 고정 길이 설정
        self.target_samples = int(self.max_duration * self.SR)

        self.split_dir = os.path.join(dataset_root, split)
        self.input_dir = os.path.join(self.split_dir, 'mixtures')
        self.s1_dir = os.path.join(self.split_dir, 'spk1')
        self.s2_dir = os.path.join(self.split_dir, 'spk2')

        if not all(os.path.exists(d) for d in [self.input_dir, self.s1_dir, self.s2_dir]):
            raise ValueError(f"Dataset directories not found in {self.split_dir}")

        all_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.wav')])

        if max_samples and len(all_files) > max_samples:
            self.file_list = all_files[:max_samples]
            print(f"Limited {split} dataset: {len(self.file_list)}/{len(all_files)} files")
        else:
            self.file_list = all_files
            print(f"Loaded {split} dataset: {len(self.file_list)} files")
        
        print(f"Target length: {self.target_samples} samples ({self.max_duration}s at {self.SR}Hz)")

    def __len__(self):
        return len(self.file_list)

    # 오디오 로드 및 고정 길이 정규화
    def _load_and_normalize_audio(self, audio_path, target_length):
        try:
            audio, _ = librosa.load(audio_path, sr=self.SR, mono=True, dtype=np.float32)
            
            # 길이 정규화
            if len(audio) < target_length:
                # 짧으면 패딩
                if len(audio) > 0:
                    repeat_times = (target_length // len(audio)) + 1
                    audio_repeated = np.tile(audio, repeat_times)
                    audio = audio_repeated[:target_length]
                else:
                    audio = np.zeros(target_length, dtype=np.float32)
            else:
                # 길면 자름
                if len(audio) > target_length:
                    start_idx = np.random.randint(0, len(audio) - target_length + 1)
                    audio = audio[start_idx:start_idx + target_length]
            
            # 최종 길이 검증 후 리턴
            if len(audio) != target_length:
                audio = np.resize(audio, target_length)
            
            return audio
            
        except Exception as e:
            print(f"Failed to load audio from {audio_path}: {e}")
            return np.zeros(target_length, dtype=np.float32)

    def __getitem__(self, idx):
        mix_filename = self.file_list[idx]  # "mix_reverb_000.wav"

        try:
            # 인덱스 추출 및 파일명 생성
            # "mix_reverb_000.wav" → "000"
            base_idx = mix_filename.split('_')[-1].replace('.wav', '')
            
            # 각 폴더의 해당 파일명 생성
            s1_filename = f"spk1_reverb_{base_idx}.wav"
            s2_filename = f"spk2_reverb_{base_idx}.wav"
            
            # 파일 경로 생성
            input_path = os.path.join(self.input_dir, mix_filename)     # mixtures/mix_reverb_000.wav
            s1_path = os.path.join(self.s1_dir, s1_filename)           # spk1/spk1_reverb_000.wav
            s2_path = os.path.join(self.s2_dir, s2_filename)           # spk2/spk2_reverb_000.wav

            # 고정 길이로 모든 오디오 로드
            input_audio = self._load_and_normalize_audio(input_path, self.target_samples)
            s1_target = self._load_and_normalize_audio(s1_path, self.target_samples)
            s2_target = self._load_and_normalize_audio(s2_path, self.target_samples)
            
            # 최종 검증
            assert len(input_audio) == self.target_samples
            assert len(s1_target) == self.target_samples
            assert len(s2_target) == self.target_samples

            return {
                'input': torch.FloatTensor(input_audio),
                'separation_targets': {
                    's1': torch.FloatTensor(s1_target),
                    's2': torch.FloatTensor(s2_target)
                },
                'filename': mix_filename  # 원본 mix 파일명 유지
            }

        except Exception as e:
            print(f"Error loading {mix_filename}: {e}")
            # 더미 데이터 (고정 길이)
            return {
                'input': torch.zeros(self.target_samples),
                'separation_targets': {
                    's1': torch.zeros(self.target_samples),
                    's2': torch.zeros(self.target_samples)
                },
                'filename': mix_filename
            }