# Deepfake Detection with EfficientNet-B3 (Face-based Binary Classifier)

이 프로젝트는 **딥페이크 영상/이미지 프레임**에서 얼굴 영역을 기반으로 Real/Fake를 분류하는  
EfficientNet-B3 이진 분류 모델입니다. InsightFace로 얼굴을 검출·크롭하고,  
캐시된 얼굴 이미지를 활용해 학습 효율과 일반화를 함께 목표로 합니다.

---

##  프로젝트 구조

```
deepfake_effnet_b3/
│
├── configs/                     # 경로 및 하이퍼파라미터 설정
│   └── config.py
├── data/                        # 데이터 전처리 및 Dataset 정의
│   ├── preprocess.py            # InsightFace 기반 얼굴 크롭 + 캐시 생성 + 증강 정의
│   └── dataset.py               # CachedFaceDataset 및 DataLoader 정의
├── models/                      # 모델 구조 (EfficientNet-B3 + EMA + FocalLoss)
│   └── efficientnet_b3.py
├── train/                       # 학습 및 유틸 함수
│   ├── train.py                 # 학습 루프 (Train / Validation / Threshold Search)
│   └── utils.py                 # 시드 고정, threshold 탐색, checkpoint 저장 등
├── infer/                       # 추론 스크립트
│   └── inference.py             # best_model.pt 기반 단일 이미지 추론
├── results/                     # 모델 및 출력물 저장 폴더
└── README.md
```

---

##  데이터 구조

학습 데이터는 Real / Fake 폴더 구조를 갖는 여러 루트를 지원합니다.  
(`configs/config.py` 의 `TRAIN_ROOTS`, `VAL_ROOTS` 설정 기준)

예시:

```
DATA_ROOT/
├── Train/
│   ├── Real/*.jpg
│   └── Fake/*.jpg
└── Validation/
    ├── Real/*.jpg
    └── Fake/*.jpg
```

---

##  얼굴 크롭 캐시 구조

학습 시 원본 이미지 대신, 미리 생성된 **얼굴 캐시**를 사용합니다:

```
CROP_CACHE_DIR/
├── Real/
│   ├── video_0001/
│   │   ├── video_0001_00030.jpg
│   │   └── ...
└── Fake/
    ├── video_1234/
    │   ├── video_1234_00030.jpg
    │   └── ...
```

캐시는 `data/preprocess.py` 실행 시 자동 생성됩니다.

---

## 🔧 1) 얼굴 캐시 생성

학습을 시작하기 전 한 번 실행합니다:

```bash
python3 data/preprocess.py
```

---

##  2) 학습 실행

```bash
python3 train/train.py
```

학습 결과는 `results/` 폴더에 저장됩니다:

```
results/
 ├── best_model.pt
 ├── best_meta.json
 ├── last_model.pt
 ├── last_full.pt
 ├── metrics.csv
 └── curves.png
```

---

##  3) 단일 이미지 추론 실행

```bash
python3 infer/inference.py --image "/path/to/image.jpg"
```

스크립트는 `best_model.pt` 와 `best_meta.json` 을 사용해  
Fake 확률과 최종 Real/Fake 결과를 출력합니다.

---
