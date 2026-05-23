# 데이터셋 상세 가이드

## 지원 데이터셋 요약

| 데이터셋 | 설정값 | 타입 | 어노테이션 |
|----------|--------|------|------------|
| GazeFollow | `gazefollow` | 정적 이미지 | gaze point (다중 annotator) |
| VideoAttentionTarget | `vat` | 비디오 | gaze point + in/out |
| ChildPlay | `childplay` | 비디오 | gaze point + LAH + LAEO + SA |
| VideoCoAtt | `videocoatt` | 비디오 | gaze point + shared attention |
| UCO-LAEO | `uco_laeo` | 비디오 | LAEO |
| VSGaze | `vsgaze` | 혼합 | 위 4개 비디오 데이터셋 통합 |

---

## VSGaze 데이터셋 (주요 학습 대상)

VSGaze는 vat + childplay + videocoatt + uco_laeo를 혼합한 복합 데이터셋.  
`VSGazeDataModule` (`mtgs/datasets/vsgaze.py`)이 4개 데이터셋을 동시에 로드한다.

각 데이터셋에서 제공하는 레이블:
- **VAT**: gaze heatmap, gaze vector, in/out
- **ChildPlay**: gaze heatmap, gaze vector, in/out, LAH, LAEO, SA
- **VideoCoAtt**: gaze heatmap, gaze vector, in/out, shared attention (SA)
- **UCO-LAEO**: LAEO 레이블 (gaze 어노테이션 품질 낮음 → 손실 0.1x 가중치 적용)

---

## 배치 데이터 구조

`mtgs/train/dataset.py`의 `build_dataset()`이 각 데이터셋 DataModule을 생성.

각 배치 샘플의 키:

```python
batch = {
    "image":        (B, T, C, H, W),         # 정규화된 장면 이미지
    "heads":        (B, T, N, C, H_h, W_h),  # 정규화된 head crop (N은 패딩 포함)
    "head_bboxes":  (B, T, N, 4),            # 정규화된 head bbox [x1,y1,x2,y2]
    "gaze_vecs":    (B, T, N, 2),            # 시선 방향 단위벡터
    "gaze_pts":     (B, T, N, 2),            # 정규화된 시선 포인트 (x, y)
    "gaze_heatmaps":(B, T, N, 64, 64),       # 가우시안 히트맵
    "inout":        (B, T, N),               # 1=inside, 0=outside, -1=unknown
    "lah_labels":   (B, T, num_pairs),       # 1=LAH, 0=not, -1=unknown
    "laeo_labels":  (B, T, num_pairs),       # 1=LAEO, 0=not, -1=unknown
    "coatt_labels": (B, T, num_pairs),       # 1=SA, 0=not, -1=unknown
    "num_valid_people": scalar,              # 실제 사람 수 (패딩 제외)
    "speaking":     (B, T, N),              # speaking 상태 (현재 미사용)
    "path": ...,                             # 이미지/비디오 경로
    "dataset": str,                          # 데이터셋 이름
}
```

**패딩 규칙**: index 0은 항상 "배경 인물"(zero tensor)로 패딩. 실제 사람은 index 1~N.

---

## 데이터 로더 설정

```python
# 학습 시 num_people = 4 (패딩 포함)
# 테스트 시 num_people = "all" (모든 사람)
num_people = {"train": 4, "val": 4, "test": "all"}

# temporal_context=2, temporal_stride=3
# → 5개 프레임을 3 프레임 간격으로 샘플링
```

---

## 어노테이션 파일 형식

모든 어노테이션은 HDF5 형식 (`.h5`). `ann_root` 디렉토리 내 위치.  
각 데이터셋 파일명은 `mtgs/datasets/` 내 각 Dataset 클래스에서 하드코딩됨.

---

## 메트릭

| 메트릭 | 설명 | 데이터셋 |
|--------|------|----------|
| AUC | 히트맵 기반 ROC AUC | 모든 데이터셋 |
| Dist | 시선 포인트 L2 거리 | 모든 데이터셋 |
| Avg/Min Dist | 평균/최솟값 annotator 거리 | GazeFollow |
| AP_IO | In/Out Average Precision | VAT, VSGaze |
| F1_LAH (PP) | LAH F1 (post-processing) | VSGaze |
| F1_LAEO (PP) | LAEO F1 (post-processing) | VSGaze |
| AP_SA | Shared Attention AP | VSGaze |

Post-processing 메트릭은 `mtgs/performance/` 의 Jupyter 노트북에서 계산.
