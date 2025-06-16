"""
[역할]
프로젝트 유틸리티 모듈
오디오 처리, 데이터 증강, 조기 종료, 경로 관리 등의 핵심 유틸리티 함수들을 제공
"""

from .audio_utils import standardize_audio_dims
from .augmentation import online_augment_sudormrf
from .early_stopping import EarlyStoppingManager
from .path_utils import (
    auto_detect_data_type,           
    setup_mixed_data_directories,    
    check_training_data,             
    check_validation_data            
)

__all__ = [
    'standardize_audio_dims',
    'online_augment_sudormrf',
    'EarlyStoppingManager',          
    'auto_detect_data_type',         
    'setup_mixed_data_directories', 
    'check_training_data',           
    'check_validation_data'          
]