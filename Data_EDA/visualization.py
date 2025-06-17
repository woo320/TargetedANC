# EDA - 시각화 파트 코드

# 시각화 위한 폰트 설치 - 한 번만 실행
# !apt-get update -qq
# !apt-get install -qq -y fonts-nanum
# !fc-cache -fv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm


# 한글 폰트 설정
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()

plt.rcParams['font.family']        = font_name
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", rc={'font.family': font_name})

# CSV 파일 경로
csv_path = "/content/drive/MyDrive/TargetedANC/Data_EDA/final_audio_eda_report.csv"

# CSV 불러오기
df = pd.read_csv(csv_path)

# 오류 있는 행 제거
if "error" in df.columns:
    df = df[df["error"].isna()]

# 시각화 기본 스타일
plt.rcParams["figure.figsize"] = (10, 6)
sns.set(style="whitegrid")

'''컬럼별 평균 및 표준 편차 확인'''
# 수치형 컬럼 선택
numeric_columns = df.select_dtypes(include="number").columns

# 평균 및 표준편차 계산
mean_values = df[numeric_columns].mean()
std_values = df[numeric_columns].std()

# 결과 통합
summary_stats = pd.DataFrame({
    "mean": mean_values,
    "std": std_values
})

# 결과 출력
print("📊 수치형 컬럼별 평균 및 표준편차:")
print(summary_stats)

''' cf_diff(중심 주파수 차이)와 stft_corr(스펙트럼 상관관계)를 기준으로
분리 난이도(이상적, 고난도) 샘플 비율 확인'''

# 이상적 조건 (cf 차이 충분 + 상관관계 낮음)
easy = df[(df["cf_diff"] >= 300) & (df["stft_corr"] <= 0.2)]

# 분리 어려운 조건 (cf 차이 매우 낮거나 상관관계 높음)
hard = df[(df["cf_diff"] < 150) | (df["stft_corr"] > 0.2)]

# 전체 개수
total = len(df)

# 결과 비율 출력
print(f"✅ 이상적 샘플 개수: {len(easy)} / {total} = {len(easy)/total:.1%}")
print(f"⚠️ 분리 어려운 샘플 개수: {len(hard)} / {total} = {len(hard)/total:.1%}")


'''시각화 코드'''
# 1. RMS 분포
rms_melted = df.melt(id_vars=["subset"], value_vars=["rms_spk1", "rms_spk2", "rms_mix"],
                     var_name="signal", value_name="RMS")
sns.boxplot(data=rms_melted, x="signal", y="RMS", hue="subset")
plt.title("RMS 에너지 분포", fontproperties=font_prop)
plt.xlabel("신호 종류", fontproperties=font_prop)
plt.ylabel("RMS", fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.show()

# 2. 중심 주파수 분포
cf_melted = df.melt(id_vars=["subset"], value_vars=["cf_spk1", "cf_spk2", "cf_mix"],
                    var_name="signal", value_name="center_freq")
sns.violinplot(data=cf_melted, x="signal", y="center_freq", hue="subset", split=True)
plt.title("중심 주파수 분포 (Hz)", fontproperties=font_prop)
plt.xlabel("신호 종류", fontproperties=font_prop)
plt.ylabel("중심 주파수 (Hz)", fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.show()

# 3. STFT 상관관계 (혼합 정도)
sns.histplot(data=df, x="stft_corr", hue="subset", bins=30, kde=True)
plt.axvline(0.2, color='g', linestyle='--', label="0.2 이상 → 혼합 강함")
plt.title("STFT 주파수 상관관계 (spk1 vs spk2)", fontproperties=font_prop)
plt.xlabel("상관계수", fontproperties=font_prop)
plt.ylabel("Count", fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.show()

# 4-1. cf_diff 분포
sns.histplot(data=df, x="cf_diff", hue="subset", bins=30, kde=True)
plt.axvline(300, color='r', linestyle='--', label="기준: 300Hz")
plt.title("spk1 vs spk2 중심 주파수 차이 (cf_diff)", fontproperties=font_prop)
plt.xlabel("Center Frequency Difference (Hz)", fontproperties=font_prop)
plt.ylabel("Count", fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.show()

# # 4-2. cf_diff vs stft_corr 상관 산점도
plt.figure(figsize=(12, 8))
palette = {"train":"#1f77b4", "val":"#ff7f0e", "test":"#2ca02c"}
sns.scatterplot(
    data=df,
    x="cf_diff",
    y="stft_corr",
    hue="subset",
    palette=palette,
    s=30,         # 마커 크기
    alpha=0.6,    # 투명도
)

# 4) 이상적/어려운 구간 시각적 강조 (배경 색칠)
ymin, ymax = df["stft_corr"].min(), df["stft_corr"].max()
xmin, xmax = df["cf_diff"].min(), df["cf_diff"].max()

# 이상적 구조 영역 (cf_diff ≥ 300, stft_corr ≤ 0.2)
plt.fill_betweenx(
    [ymin, 0.2],
    300, xmax,
    color="green",
    alpha=0.2,
    label="이상적 구조 영역"
)
# (수정) 고난도 구조: cf_diff < 150 AND stft_corr > 0.2
plt.fill_betweenx(
    [0.2, ymax],   # y: 0.2 초과 구간
    xmin, 150,     # x: 0 ~ 150Hz 구간
    color='indianred',
    alpha=0.2,
    label='고난이도 구조'
)

# 5) 기준선 & 레이블
plt.axvline(300, color='green', linestyle='--', linewidth=2)
plt.axhline(0.2, color='red', linestyle='--', linewidth=2)
plt.text(800, 0.15,
         "이상적 구조",
         fontproperties=font_prop,
         fontsize=16,
         color="darkgreen",
         ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="darkgreen", alpha=0.8))

plt.text(77, 0.325,
         "고난도 구조",
         fontproperties=font_prop,
         fontsize=16,
         color="darkred",
         ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="darkred", alpha=0.8))


# 6) 제목·축·범례 설정
plt.xlim(0, 1600)
plt.ylim(0, ymax)
plt.title("주파수 차이 vs 상관관계 (분리 난이도 시각화)", fontproperties=font_prop, size=18)
plt.xlabel("cf_diff (Hz)", fontproperties=font_prop, size=14)
plt.ylabel("STFT corr", fontproperties=font_prop, size=14)
plt.legend(prop=font_prop, fontsize=12, loc="upper right")
plt.tight_layout()
plt.show()