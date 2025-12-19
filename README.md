# Drone Image Instance Segmentation (Mask R-CNN)

이 프로젝트는 **Mask R-CNN** 딥러닝 모델을 활용하여 드론으로 촬영된 항공 이미지 내의 객체(사람, 차량 등)를 탐지하고 분할(Instance Segmentation)하는 모델을 학습시키는 파이프라인입니다.

## Project Overview
드론 이미지는 객체의 크기가 작고, 배경이 복잡하며, 다양한 각도에서 촬영된다는 특징이 있습니다. 본 프로젝트는 이러한 특성을 고려하여 **PyTorch** 기반의 Mask R-CNN 모델을 구축하고, 커스텀 데이터셋(COCO Format)을 로드하여 학습 및 추론을 수행합니다.

### Key Features
* **Custom Dataset Loading**: COCO 포맷의 JSON 어노테이션을 파싱하여 Mask R-CNN 입력 형식에 맞게 변환합니다.
* **Transfer Learning**: `COCO_V1`으로 사전 학습된 가중치를 사용하여 적은 데이터로도 빠른 수렴을 유도합니다.
* **Data Augmentation**: 학습 시 `RandomHorizontalFlip` 등을 적용하여 모델의 일반화 성능을 높입니다.
* **Validation Loop**: 학습 중 검증 데이터셋(Validation Set)에 대한 Loss를 모니터링하여 과적합을 방지합니다.
* **Inference Visualization**: 학습된 모델이 예측한 마스크(Segmentation Mask)를 시각화하여 성능을 직관적으로 확인합니다.

## 🛠️ Tech Stack & Environment

이 프로젝트는 다음의 기술과 라이브러리를 기반으로 작성되었습니다.

| Category | Technology | Description |
| :--- | :--- | :--- |
| **Framework** | **PyTorch** | 딥러닝 모델 구현 및 학습 |
| **Library** | **Torchvision** | 사전 학습된 모델(Backbone) 및 이미지 변환 도구 제공 |
| **Model Arch** | **Mask R-CNN** | ResNet-50 Backbone + FPN (Feature Pyramid Network) 구조 사용 |
| **Data Processing** | **PyCOCOTools** | COCO 포맷의 JSON 데이터 파싱 및 마스크 디코딩 |
| **Visualization** | **Matplotlib** | 이미지 및 예측 마스크 시각화 |
| **Utils** | **NumPy, PIL, Tqdm** | 행렬 연산, 이미지 로드, 학습 진행률 표시 |

## 📂 Code Structure & Functions

주요 코드와 함수들에 대한 설명입니다.

### 1. Data Processing (`DroneCocoDataset`, `polygon_to_mask`)
* **`polygon_to_mask(segmentation, height, width)`**:
    * JSON 파일 내의 다각형(Polygon) 좌표 데이터를 모델이 이해할 수 있는 비트맵(Bitmap) 형태의 마스크로 변환합니다.
    * `pycocotools.mask` 모듈을 사용하여 RLE 디코딩을 수행합니다.
* **`class DroneCocoDataset(Dataset)`**:
    * PyTorch의 `Dataset` 클래스를 상속받아 구현되었습니다.
    * 이미지 경로와 어노테이션을 매핑하고, 학습 시 이미지와 타겟(BBox, Label, Mask)을 텐서로 반환합니다.

### 2. Transforms & Augmentation (`get_transform`)
* **`get_transform(train)`**:
    * 이미지 데이터를 텐서로 변환합니다.
    * `train=True`일 경우, `RandomHorizontalFlip`(좌우 반전) 등의 증강 기법을 적용하여 데이터의 다양성을 확보합니다.

### 3. Model Definition
* **`maskrcnn_resnet50_fpn(weights="COCO_V1")`**:
    * ResNet-50을 백본으로 사용하고 FPN이 적용된 Mask R-CNN 모델을 불러옵니다.
    * `FastRCNNPredictor`와 `MaskRCNNPredictor`의 Head 부분을 교체하여 사용자 정의 클래스 개수(num_classes)에 맞게 미세 조정(Fine-tuning)합니다.

### 4. Training & Validation
* **`validate_loss(model, data_loader, device)`**:
    * 검증(Validation) 단계에서 모델의 가중치를 업데이트하지 않고(`torch.no_grad`), 순수하게 Loss만 계산하여 과적합 여부를 판단합니다.
* **Training Loop**:
    * SGD Optimizer와 StepLR 스케줄러를 사용하여 학습을 진행합니다.
    * Epoch마다 Checkpoint를 저장하고, Validation Loss가 가장 낮은 모델을 `best_model.pth`로 저장합니다.

## How to Run

1.  **데이터셋 준비**:
    * `IMG_DIR` 경로에 이미지 파일들을 위치시킵니다.
    * `JSON_PATH` 경로에 COCO 포맷의 어노테이션 파일을 위치시킵니다.

2.  **경로 설정**:
    * 코드 상단의 경로 변수(`IMG_DIR`, `JSON_PATH`, `CHECKPOINT_DIR`)를 본인의 환경에 맞게 수정합니다.

3.  **학습 실행**:
    ```bash
    python gyeol_train.py  # 또는 해당 스크립트 실행
    ```

## Results

gyeol_predict.py 모델이 새로운 이미지에 대한 예측한 마스크를 시각화합니다.

---
