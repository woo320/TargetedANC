from collections import defaultdict

import numpy as np

class EarlyStoppingManager:
    """다중 메트릭 기반 조기 종료 관리 (기존 인터페이스 유지)"""

    def __init__(self, patience, min_delta, metric_weights=None):
        self.patience = patience
        self.min_delta = min_delta
        
        # 🔧 다중 메트릭 가중치 설정
        self.metric_weights = metric_weights or {
            'anc_total': 0.4,           # ANC 성능 40%
            'separation_loss': 0.3,     # 분리 성능 30%
            'classification_accuracy': 0.2,  # 분류 성능 20%
            'final_quality': 0.1        # 최종 품질 10%
        }
        
        # 상태 관리
        self.metrics_history = defaultdict(list)
        self.best_composite_score = -float('inf')
        self.best_scores = {}  # 개별 메트릭 베스트 점수
        self.no_improvement_count = 0
        
        print(f"🔧 MultiMetric EarlyStopping initialized:")
        print(f"   Patience: {patience}, Min Delta: {min_delta}")
        print(f"   Weights: {self.metric_weights}")

    def compute_composite_score(self, metrics):
        """여러 메트릭을 조합한 종합 점수 계산"""
        score = 0.0
        score_details = {}
        
        # ANC 점수 (낮을수록 좋음, 음수값이므로 부호 조정)
        anc_loss = metrics.get('anc_total', 0)
        anc_score = -anc_loss * self.metric_weights['anc_total']  # 음수를 양수로
        score += anc_score
        score_details['anc_score'] = anc_score
        
        # 분리 손실 (낮을수록 좋음)
        sep_loss = metrics.get('separation_loss', 
                              (metrics.get('s1_separation', 0) + metrics.get('s2_separation', 0)) / 2)
        sep_score = -sep_loss * self.metric_weights['separation_loss']
        score += sep_score
        score_details['separation_score'] = sep_score
        
        # 분류 정확도 (높을수록 좋음)
        cls_acc = metrics.get('classification_accuracy', 0)
        cls_score = cls_acc * self.metric_weights['classification_accuracy']
        score += cls_score
        score_details['classification_score'] = cls_score
        
        # 최종 품질 (낮을수록 좋음)
        final_loss = metrics.get('final_quality', 0)
        final_score = -final_loss * self.metric_weights['final_quality']
        score += final_score
        score_details['final_quality_score'] = final_score
        
        return score, score_details

    def update(self, metrics):
        """메트릭 업데이트"""
        # 기존 호환성을 위한 히스토리 저장
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        # 종합 점수 계산
        current_score, score_details = self.compute_composite_score(metrics)
        
        # 개별 메트릭 베스트 업데이트 (로깅용)
        for metric_name, value in metrics.items():
            if metric_name not in self.best_scores:
                self.best_scores[metric_name] = value
            elif metric_name in ['anc_total', 'separation_loss', 'final_quality']:
                # 낮을수록 좋은 메트릭
                if value < self.best_scores[metric_name]:
                    self.best_scores[metric_name] = value
            elif metric_name == 'classification_accuracy':
                # 높을수록 좋은 메트릭
                if value > self.best_scores[metric_name]:
                    self.best_scores[metric_name] = value
        
        # 종합 점수 기반 조기 종료 판단
        improvement = current_score - self.best_composite_score
        
        if improvement > self.min_delta:
            self.best_composite_score = current_score
            self.no_improvement_count = 0
            
            # 상세 정보 출력 (개선 시에만)
            print(f"🎯 Early Stopping: NEW BEST composite score: {current_score:.4f}")
            print(f"   📊 Score breakdown: ANC={score_details['anc_score']:.3f}, "
                  f"Sep={score_details['separation_score']:.3f}, "
                  f"Cls={score_details['classification_score']:.3f}, "
                  f"Qual={score_details['final_quality_score']:.3f}")
            
        else:
            self.no_improvement_count += 1
            
            # 개선 없음 경고 (가끔씩만)
            if self.no_improvement_count % 2 == 0:
                print(f"⏳ Early Stopping: No improvement for {self.no_improvement_count}/{self.patience} epochs")

    def should_stop(self):
        """조기 종료 여부 결정"""
        return self.no_improvement_count >= self.patience
    
    def get_best_scores(self):
        """베스트 점수들 반환 (호환성)"""
        return {
            'composite_score': self.best_composite_score,
            'individual_bests': self.best_scores.copy()
        }
    
    def get_improvement_summary(self):
        """개선 상황 요약"""
        if not self.metrics_history:
            return "No metrics recorded yet"
        
        summary = []
        for metric_name in ['anc_total', 'separation_loss', 'classification_accuracy', 'final_quality']:
            if metric_name in self.metrics_history and len(self.metrics_history[metric_name]) >= 2:
                values = self.metrics_history[metric_name]
                initial = values[0]
                current = values[-1]
                
                if metric_name == 'classification_accuracy':
                    change = current - initial
                    direction = "↑" if change > 0 else "↓"
                    summary.append(f"{metric_name}: {initial:.3f} → {current:.3f} {direction}")
                else:
                    change = initial - current  # 낮을수록 좋은 메트릭
                    direction = "↓" if change > 0 else "↑"  
                    summary.append(f"{metric_name}: {initial:.3f} → {current:.3f} {direction}")
        
        return " | ".join(summary)
