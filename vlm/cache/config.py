"""내 config.yaml 을 로드해 VSGaze offline 추출/평가용 cfg 를 만드는 헬퍼.
peer 의 하드코딩 ROOT(/home/sujungoh/...) 대신 이 리포의 경로를 그대로 쓴다."""
from pathlib import Path
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent          # .../MTGS
CONFIG_PATH = REPO_ROOT / "mtgs" / "config" / "config.yaml"
QWEN = "Qwen/Qwen3-VL-8B-Instruct"


def make_cfg(split, *, task="test", use_graph=True, num_people=None):
    """config.yaml 을 로드해 VSGaze <split> 를 test 파이프라인으로 돌리는 cfg 반환.
    num_people 를 주면 data.num_people 를 덮어써 offline 추출을 N=all 로 강제할 수 있다
    (그래프 '학습' 기본값 config.data.num_people 는 건드리지 않는다)."""
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.set_struct(cfg, False)
    cfg.device = "cuda"
    cfg.wandb.log = False
    cfg.experiment.dataset = "vsgaze"
    cfg.experiment.task = task
    cfg.data.temporal_context = 2
    cfg.gaze_graph.use = use_graph
    cfg.data.test_split = split
    cfg.model.weights = False
    if num_people is not None:
        cfg.data.num_people = num_people
    return cfg
