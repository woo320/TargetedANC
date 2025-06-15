import torch
import gc
import psutil
from config.constants import SR

class MemoryManager:

    def __init__(self, config):
        self.max_memory_gb = config.get('max_memory_gb', 22.5)
        self.memory_warning_threshold = config.get('memory_warning_threshold', 20.0)
        self.memory_critical_threshold = config.get('memory_critical_threshold', 21.5)
        self.adaptive_chunk_sizes = config.get('adaptive_chunk_sizes', {
            'low_memory': 1.0 * 16000,
            'medium_memory': 1.5 * 16000,
            'normal_memory': 2.0 * 16000,
            'high_memory': 3.0 * 16000
        })
        self.memory_history = []

    def get_memory_usage(self):
        """현재 메모리 사용량 반환 (GB)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return allocated, reserved
        return 0, 0
    
    def get_safe_chunk_size(self):
        """현재 메모리 상황에 따른 안전한 청크 크기 반환"""
        allocated, reserved = self.get_memory_usage()

        if allocated > self.memory_critical_threshold:
            return int(self.adaptive_chunk_sizes['low_memory'])
        elif allocated > self.memory_warning_threshold:
            return int(self.adaptive_chunk_sizes['medium_memory'])
        elif allocated > 15:
            return int(self.adaptive_chunk_sizes['normal_memory'])
        else:
            return int(self.adaptive_chunk_sizes['high_memory'])

    def is_memory_critical(self):
        """메모리가 위험 수준인지 확인"""
        allocated, _ = self.get_memory_usage()
        return allocated > self.memory_critical_threshold

    def get_system_memory(self):
        """시스템 메모리 사용량"""
        return psutil.virtual_memory().used / 1e9

    def cleanup_memory(self, aggressive=False):
        """메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        if aggressive:
            for _ in range(3):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def adaptive_batch_size(self, default_batch_size=1):
        allocated, _ = self.get_memory_usage()
        
        if allocated > self.memory_critical_threshold:   
            return 1
        elif allocated > self.memory_warning_threshold:  
            # 🔧 안전한 절반 계산
            if default_batch_size <= 1:
                return 1
            else:
                return max(1, default_batch_size // 2)
        elif allocated < 10.0:  # 메모리 여유 (10GB 미만 사용)
            # 🔧 효과적인 배치 증가
            return min(max(default_batch_size * 2, 4), 8)
        else:
            return default_batch_size