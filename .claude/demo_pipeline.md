# 데모 파이프라인 상세

## 전체 흐름

```
비디오 입력
    ↓
[HeadDetector] YOLOv5-CrowdHuman
    → raw detections (bbox, confidence)
    ↓
[Tracker] OCSORT (boxmot)
    → 사람별 person_id + tracked bbox
    ↓
[DemoProcessor.prepare_input()]
    → image 정규화, head crop 추출, bbox 정규화
    ↓
[GazePredictor] MTGS 모델
    → gaze_heatmap, gaze_vec, inout, lah, laeo, coatt
    ↓
[get_social_gaze_predictions()] 후처리
    → 기하학적 LAH/LAEO 재계산 (gaze point → head bbox 내 포함 여부)
    ↓
[draw_gaze()] 시각화
    → 오버레이된 프레임
    ↓
CSV 저장 + MP4 저장
```

---

## 주요 모듈 (mtgs/demo/)

### HeadDetector (`head_detection.py`)
- YOLOv5-CrowdHuman 기반 헤드 검출
- 입력: 원본 프레임 (BGR → RGB 변환 후)
- `detection_thr=0.4`, `conf_thr=0.25`, `iou_thr=0.45`
- `expand_bbox=0.1`: 검출된 bbox를 10% 확장

### Tracker (`tracking.py`)
- OCSORT (boxmot 라이브러리) 사용
- 프레임 간 person_id 유지
- `max_age=300`: 300 프레임 동안 트랙 유지

### GazePredictor (`gaze_prediction.py`)
- MTGS 체크포인트 로드 및 추론
- temporal_context=0 (static) 모드가 데모 기본값

---

## 입력 준비 (prepare_input)

```python
# head crop: square bbox로 변환 후 224×224로 resize
heads = [resize(crop(image, bbox), 224) for bbox in head_bboxes]

# 이미지: 448×448로 resize, 정규화
image = normalize(resize(image, 448))

# 패딩: index 0은 zero tensor (배경 인물)
heads = cat([zeros(1,3,224,224), heads])        # (N+1, 3, 224, 224)
head_bboxes = cat([zeros(1,4), head_bboxes])    # (N+1, 4)

# batch/temporal 차원 추가
sample["image"]      = image.unsqueeze(0).unsqueeze(0)      # (1, 1, C, H, W)
sample["heads"]      = heads.unsqueeze(0).unsqueeze(0)      # (1, 1, N+1, C, H_h, W_h)
sample["head_bboxes"] = head_bboxes.unsqueeze(0).unsqueeze(0) # (1, 1, N+1, 4)
```

---

## 소셜 게이즈 후처리 (`utils/social_gaze.py`)

데모에서는 MTGS 모델의 LAH/LAEO 예측 대신 **기하학적 후처리**를 사용:

```python
# person2가 person1의 head bbox 안을 보고 있으면 LAH
if is_inside(head_bbox1, gaze_pred2):
    lah[pair] = 1
    # 서로 보면 LAEO
    if is_inside(head_bbox2, gaze_pred1):
        laeo[pair] = 1
```

SA(coatt)는 모델 예측값을 그대로 사용.

---

## 출력 파일

| 파일 | 설명 |
|------|------|
| `{filename}-pred.mp4` | 시각화된 비디오 |
| `{filename}-pred.csv` | 프레임별 예측 결과 (pandas DataFrame) |

CSV 컬럼: `frame_nb`, `gaze_pt_x`, `gaze_pt_y`, `gaze_vec_x`, `gaze_vec_y`, `inout`, `lah_id`, `laeo_id`, `coatt_id`, `pid`, `xmin`, `ymin`, `xmax`, `ymax`

---

## 데모 실행 명령

```bash
cd scripts

python ./demo.py \
    head_detector.checkpoint_file=/path/to/yolov5_crowdhuman.pt \
    demo.video_file=/path/to/video.mp4 \
    demo.checkpoint_file=/path/to/mtgs-static-vsgaze.ckpt \
    demo.output_folder=/path/to/output/ \
    data.temporal_context=0 \
    demo.heatmap_pid=-1   # -1: 모든 사람, 0+: 특정 pid의 히트맵만 표시
```

---

## 주의사항

- 데모는 **static model** (`temporal_context=0`) 권장 — 프레임 단위 처리이기 때문
- temporal 모델 사용 시 `data.temporal_context=2` 로 설정하면 window 버퍼링이 필요
- YOLO 체크포인트는 CrowdHuman으로 학습된 YOLOv5 모델 필요 (일반 YOLOv5와 다름)
- 출력 mp4는 ffmpeg 기반 (시스템에 ffmpeg 설치 필요)
