"""
ASC 모델 평가 스크립트
- 이 스크립트는 ASC 모델을 학습된 가중치(.pth)를 이용하여 테스트 데이터셋(안내방송/소음 폴더)에 대해 평가를 수행
- DataLoader와 sklearn metrics를 활용하여 손실, 정확도, 정밀도, 재현율, F1, ROC-AUC, PR-AUC, 혼동행렬, 분류 리포트 등 주요 지표를 산출
- 추론 구간만을 별도로 측정하여 1초 오디오 처리 기준 ms/sec도 함께 출력
"""

import os
import argparse
import torch
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from ASC import ASCDataset, ASC

def evaluate(model, loader, device, sr=16000, window_len=16000):
    """
    모델 평가 함수

    - 모델을 eval 모드로 전환, 배치별로 추론 후 주요 분류 지표 산출
    - 첫 배치 진입 시점부터 마지막 배치까지의 추론 구간 시간만 별도 측정
    - 전체 오디오 길이 대비 실질적인 처리속도(ms/sec) 산출
    """
    model.eval()
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_windows = 0

    start_time = None  # 추론 타이밍 측정 시작점

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            if start_time is None:
                start_time = time.time()  # 첫 배치 진입 시점부터 측정 시작

            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            loss = criterion(logits, yb).item() * xb.size(0)
            total_loss += loss

            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            n_windows += xb.size(0)

    elapsed = time.time() - start_time  # 데이터 로딩 제외, 추론 구간만 측정

    n = len(loader.dataset)
    avg_loss = total_loss / n
    acc = ( (torch.tensor(all_preds) == torch.tensor(all_labels)) .sum().item() ) / n
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    roc  = roc_auc_score(all_labels, all_probs)
    pr   = average_precision_score(all_labels, all_probs)
    cm   = confusion_matrix(all_labels, all_preds)

    total_samples = n * window_len
    audio_sec = total_samples / sr
    ms_per_sec = elapsed / audio_sec * 1000  # 1초 오디오 처리 시간(ms)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}, PR-AUC: {pr:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Noise","Broadcast"]))
    print("------")
    print(f"총 추론 시간(데이터 로딩 제외): {elapsed:.3f} sec")
    print(f"평가한 전체 오디오 길이: {audio_sec:.2f} sec ({total_samples} samples)")
    print(f"1초 오디오 처리 시간: {ms_per_sec:.2f} ms/sec")

def main():
    """
    메인 함수: 인자 파싱 → 데이터셋 및 DataLoader 준비 → 모델 로드 → 평가 실행
    """
    parser = argparse.ArgumentParser(description="Evaluate ASC")
    parser.add_argument('--test_s1_dir',   required=True, help="테스트용 broadcast 폴더 경로")
    parser.add_argument('--test_s2_dir',   required=True, help="테스트용 noise 폴더 경로")
    parser.add_argument('--model_path',    required=True, help=".pth 모델 파일 경로")
    parser.add_argument('--batch_size',    type=int, default=16)
    parser.add_argument('--sr',            type=int, default=16000)
    parser.add_argument('--window_len',    type=int, default=16000)
    parser.add_argument('--max_workers',   type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 & DataLoader
    test_ds = ASCDataset(
        broadcast_dir=args.test_s1_dir,
        noise_dir=args.test_s2_dir,
        sr=args.sr,
        window_len=args.window_len,
        max_workers=args.max_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.max_workers,
        pin_memory=True
    )

    # 모델 정의 및 가중치 로드
    model = ASC(window_len=args.window_len).to(device)
    state = torch.load(args.model_path, map_location=device)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)

    print("Evaluation 시작")
    evaluate(model, test_loader, device, sr=args.sr, window_len=args.window_len)

if __name__ == '__main__':
    main()
