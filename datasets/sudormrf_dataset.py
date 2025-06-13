import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from config.constants import SR

class SudoRMRFDynamicMixDataset(Dataset):
    """SudoRM-RF 방식의 실시간 s1 + s2 믹싱 데이터셋 (길이 일관성 보장)"""
    def __init__(self, dataset_root, split='train', max_samples=None, max_duration=15.0,
                 use_online_augment=True, config=None):
        self.dataset_root = dataset_root
        self.split = split
        self.max_duration = max_duration
        self.use_online_augment = use_online_augment
        
        # config에서 sample_rate 가져오기
        if config is not None:
            self.SR = config.get('sample_rate', 16000)
        else:
            # 하위호환성을 위한 fallback
            from config.constants import DEFAULT_SR
            self.SR = DEFAULT_SR
        
        # 🔧 고정 길이 설정 (일관성 보장)
        self.target_samples = int(self.max_duration * self.SR)

        self.split_dir = os.path.join(dataset_root, split)
        self.s1_dir = os.path.join(self.split_dir, 'spk1')
        self.s2_dir = os.path.join(self.split_dir, 'spk2')

        if not all(os.path.exists(d) for d in [self.s1_dir, self.s2_dir]):
            raise ValueError(f"SudoRM-RF DynamicMix directories not found: {self.s1_dir}, {self.s2_dir}")

        # s1과 s2 파일 목록
        self.s1_files = sorted([f for f in os.listdir(self.s1_dir) if f.endswith('.wav')])
        self.s2_files = sorted([f for f in os.listdir(self.s2_dir) if f.endswith('.wav')])

        if not self.s1_files:
            raise ValueError(f"No s1 files found in {self.s1_dir}")
        if not self.s2_files:
            raise ValueError(f"No s2 files found in {self.s2_dir}")

        min_files = min(len(self.s1_files), len(self.s2_files))

        if max_samples and min_files > max_samples:
            self.s1_files = self.s1_files[:max_samples]
            self.s2_files = self.s2_files[:max_samples]
            min_files = max_samples

        self.num_samples = min_files
        print(f"📂 SudoRM-RF DynamicMix {split} dataset: {self.num_samples} pairs")
        print(f"   Target length: {self.target_samples} samples ({self.max_duration}s at {self.SR}Hz)")
        print(f"   S1 files: {len(self.s1_files)}, S2 files: {len(self.s2_files)}")
        print(f"   Online augmentation: {use_online_augment}")

    def __len__(self):
        return self.num_samples

    def _load_and_normalize_audio(self, audio_path, target_length):
        """오디오 로드 및 고정 길이로 정규화"""
        try:
            audio, _ = librosa.load(audio_path, sr=self.SR, mono=True, dtype=np.float32)
            
            # 🔧 길이 정규화: 패딩 또는 자르기
            if len(audio) < target_length:
                # 짧으면 패딩 (반복 패딩으로 더 자연스럽게)
                if len(audio) > 0:
                    repeat_times = (target_length // len(audio)) + 1
                    audio_repeated = np.tile(audio, repeat_times)
                    audio = audio_repeated[:target_length]
                else:
                    audio = np.zeros(target_length, dtype=np.float32)
            else:
                # 길면 랜덤 크롭 (학습 데이터 다양성 증가)
                if len(audio) > target_length:
                    start_idx = np.random.randint(0, len(audio) - target_length + 1)
                    audio = audio[start_idx:start_idx + target_length]
            
            # 🔧 최종 길이 검증
            if len(audio) != target_length:
                print(f"⚠️ Length mismatch after processing: {len(audio)} vs {target_length}")
                audio = np.resize(audio, target_length)  # 강제 리사이즈
            
            return audio
            
        except Exception as e:
            print(f"❌ Failed to load audio from {audio_path}: {e}")
            return np.zeros(target_length, dtype=np.float32)

    def __getitem__(self, idx):
        try:
            # s1과 s2 파일 경로
            s1_filename = self.s1_files[idx % len(self.s1_files)]
            s2_filename = self.s2_files[idx % len(self.s2_files)]
            
            s1_path = os.path.join(self.s1_dir, s1_filename)
            s2_path = os.path.join(self.s2_dir, s2_filename)

            # 🔧 고정 길이로 오디오 로드
            s1_audio = self._load_and_normalize_audio(s1_path, self.target_samples)
            s2_audio = self._load_and_normalize_audio(s2_path, self.target_samples)
            
            # 🔧 최종 검증
            assert len(s1_audio) == self.target_samples, f"S1 length mismatch: {len(s1_audio)} vs {self.target_samples}"
            assert len(s2_audio) == self.target_samples, f"S2 length mismatch: {len(s2_audio)} vs {self.target_samples}"

            # SudoRM-RF 방식: 개별 소스들을 스택으로 제공
            sources = np.stack([s1_audio, s2_audio], axis=0)  # [2, T]

            return {
                'sources': torch.FloatTensor(sources),  # [2, T] - s1, s2 개별 소스
                'separation_targets': {
                    's1': torch.FloatTensor(s1_audio),
                    's2': torch.FloatTensor(s2_audio)
                },
                's1_filename': s1_filename,
                's2_filename': s2_filename
            }

        except Exception as e:
            print(f"❌ Error creating SudoRM-RF mix for index {idx}: {e}")
            # 더미 데이터 (고정 길이)
            dummy_sources = np.zeros((2, self.target_samples), dtype=np.float32)
            return {
                'sources': torch.FloatTensor(dummy_sources),
                'separation_targets': {
                    's1': torch.zeros(self.target_samples),
                    's2': torch.zeros(self.target_samples)
                },
                's1_filename': f"dummy_s1_{idx}.wav",
                's2_filename': f"dummy_s2_{idx}.wav"
            }