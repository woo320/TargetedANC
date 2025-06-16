from collections import defaultdict
import numpy as np

class EarlyStoppingManager:
    """
    [역할]
    다중 메트릭 기반 Early Stopping Manager
    여러 성능 지표를 종합적으로 고려하여 조기 종료 여부를 결정
    """

    def __init__(self, patience, min_delta, metric_weights=None):

        self.patience = patience
        self.min_delta = min_delta
        
        # 다중 메트릭 가중치 기본값 설정
        self.metric_weights = metric_weights or {
            'anc_total': 0.4,                
            'separation_loss': 0.3,          
            'classification_accuracy': 0.2,  
            'final_quality': 0.1             
        }
        
        self.metrics_history = defaultdict(list)   
        self.best_composite_score = -float('inf')  
        self.best_scores = {}                      
        self.no_improvement_count = 0              
        
        print(f"🔧 다중 메트릭 조기 종료 관리자 초기화:")
        print(f"   인내심: {patience}, 최소 변화량: {min_delta}")
        print(f"   가중치: {self.metric_weights}")

    def compute_composite_score(self, metrics):
        """
        [역할] 여러 메트릭을 조합해 종합 점수 계산
        """
        
        score = 0.0
        score_details = {}
        
        # 1. ANC 점수
        anc_loss = metrics.get('anc_total', 0)
        anc_score = -anc_loss * self.metric_weights['anc_total']
        score += anc_score
        score_details['anc_score'] = anc_score
        
        # 2. 분리 점수
        sep_loss = metrics.get('separation_loss', 
                              (metrics.get('s1_separation', 0) + metrics.get('s2_separation', 0)) / 2)
        sep_score = -sep_loss * self.metric_weights['separation_loss']
        score += sep_score
        score_details['separation_score'] = sep_score
        
        # 3. 분류 점수
        cls_acc = metrics.get('classification_accuracy', 0)
        cls_score = cls_acc * self.metric_weights['classification_accuracy']
        score += cls_score
        score_details['classification_score'] = cls_score
        
        # 4. 최종 점수
        final_loss = metrics.get('final_quality', 0)
        final_score = -final_loss * self.metric_weights['final_quality']
        score += final_score
        score_details['final_quality_score'] = final_score
        
        return score, score_details

    def update(self, metrics):
        """
        [역할]
        새로운 에포크의 메트릭으로 조기 종료 상태 업데이트
        종합 점수 기반 개선 판단
        """

        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        current_score, score_details = self.compute_composite_score(metrics)
        
        for metric_name, value in metrics.items():
            if metric_name not in self.best_scores:
                self.best_scores[metric_name] = value

            elif metric_name in ['anc_total', 'separation_loss', 'final_quality']:
                if value < self.best_scores[metric_name]:
                    self.best_scores[metric_name] = value

            elif metric_name == 'classification_accuracy':
                if value > self.best_scores[metric_name]:
                    self.best_scores[metric_name] = value
        
        improvement = current_score - self.best_composite_score
        
        if improvement > self.min_delta:
            self.best_composite_score = current_score
            self.no_improvement_count = 0
            
            print(f"  조기 종료: 새로운 최고 종합 점수: {current_score:.4f}")
            print(f"  점수 세부사항: ANC={score_details['anc_score']:.3f}, "
                  f"분리={score_details['separation_score']:.3f}, "
                  f"분류={score_details['classification_score']:.3f}, "
                  f"품질={score_details['final_quality_score']:.3f}")
            
        else:
            self.no_improvement_count += 1
            
            if self.no_improvement_count % 2 == 0:
                print(f"  조기 종료: {self.no_improvement_count}/{self.patience} 에포크 동안 개선 없음")

    def should_stop(self):
        """
        [역할] 조기 종료 여부 결정
        """
        return self.no_improvement_count >= self.patience
    
    def get_best_scores(self):
        """
        [역할] 종합 점수와 개별 최고 점수 반환 (호환성 유지)
        """
        return {
            'composite_score': self.best_composite_score,
            'individual_bests': self.best_scores.copy()
        }
    
    def get_improvement_summary(self):
        """
        [역할]
        학습 과정에서의 개선 상황 요약
        각 메트릭 초기값 vs 현재값 비교 후 어떻게 변화했는지 표시
        """

        if not self.metrics_history:
            return "아직 기록된 메트릭이 없습니다"
        
        summary = []
        
        for metric_name in ['anc_total', 'separation_loss', 'classification_accuracy', 'final_quality']:
            if metric_name in self.metrics_history and len(self.metrics_history[metric_name]) >= 2:
                values = self.metrics_history[metric_name]
                initial = values[0]    
                current = values[-1]
                
                if metric_name == 'classification_accuracy':
                    # 정확도
                    change = current - initial
                    direction = "↑" if change > 0 else "↓"
                    summary.append(f"{metric_name}: {initial:.3f} → {current:.3f} {direction}")
                else:
                    # 손실
                    change = initial - current
                    direction = "↓" if change > 0 else "↑"  
                    summary.append(f"{metric_name}: {initial:.3f} → {current:.3f} {direction}")
        
        return " | ".join(summary)