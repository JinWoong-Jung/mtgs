"""내 config.yaml 을 로드해 VSGaze offline 추출/평가용 cfg 를 만드는 헬퍼.
peer 의 하드코딩 ROOT(/home/sujungoh/...) 대신 이 리포의 경로를 그대로 쓴다."""
from pathlib import Path
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent          # .../MTGS
CONFIG_PATH = REPO_ROOT / "mtgs" / "config" / "config.yaml"
QWEN = "Qwen/Qwen3-VL-8B-Instruct"


def make_cfg(split, *, task="test", use_graph=True):
    """config.yaml 을 로드해 VSGaze <split> 를 test 파이프라인으로 돌리는 cfg 반환.
    데이터/체크포인트 경로는 config.yaml 값을 그대로 사용(추가 하드코딩 없음)."""
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.set_struct(cfg, False)
    cfg.device = "cuda"
    cfg.wandb.log = False
    cfg.experiment.dataset = "vsgaze"
    cfg.experiment.task = task
    cfg.data.temporal_context = 2
    cfg.gaze_graph.use = use_graph
    # split 선택: 내 datamodule 이 test_split 을 참조하면 세팅(없으면 무시됨)
    cfg.data.test_split = split
    cfg.model.weights = False   # ckpt 는 graph_export 에서 직접 load_state_dict
    return cfg
