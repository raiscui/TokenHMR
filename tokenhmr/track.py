import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
# import argparse

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class TokenHMRPredictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from lib.models import load_tokenhmr

        # Load checkpoints
        model, _ = load_tokenhmr(checkpoint_path=cfg.checkpoint, \
                                 model_cfg=cfg.model_config, \
                                 is_train_state=False, is_demo=True)

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)

        # Overriding the SMPL params with the TokenHMR params
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

class PHALP_Prime_TokenHMR(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def cached_download_from_drive(self, additional_urls=None):
        """
        覆盖 PHALP 默认的资源下载逻辑.

        背景:
        - TokenHMR demo 使用的 PHALP fork(`saidwivedi/PHALP`)内部默认从 Berkeley 站点下载资源.
        - 该站点当前对外网环境会返回 "Access restricted/403",导致 `demo_video` 无法运行.
        - 我们这里把资源下载链接切换到 UT Austin 的镜像,并保持 PHALP 的缓存目录结构不变.

        说明:
        - 该方法会在 `PHALP.__init__()` 内被调用.
        - `additional_urls` 由上游保留,用于额外扩展下载项.
        """
        import os
        import shutil

        from phalp.configs.base import CACHE_DIR
        from phalp.utils.utils_download import cache_url

        # ---------------------------------------------------------------------
        # 1) 先确保 PHALP 期望的缓存目录存在.
        # ---------------------------------------------------------------------
        os.makedirs(os.path.join(CACHE_DIR, "phalp"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/3D/models/smpl"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/weights"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/ava"), exist_ok=True)

        # ---------------------------------------------------------------------
        # 2) 能从 TokenHMR 的 `data/body_models` 复用的文件,优先本地复制,减少下载且更稳定.
        #    如果这些文件缺失,通常说明还没执行 `pixi run fetch_demo_data`.
        # ---------------------------------------------------------------------
        # 注意: Hydra 默认会把当前工作目录切到 `outputs/...`,
        # 因此这里不能用 os.getcwd() 拼相对路径.
        repo_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        local_body_models_dir = os.path.join(repo_root_dir, "data/body_models")
        local_smpl_dir = os.path.join(local_body_models_dir, "smpl")

        # SMPL 模型: PHALP 默认读取 ~/.cache/phalp/3D/models/smpl/*.pkl
        smpl_cache_dir = os.path.join(CACHE_DIR, "phalp/3D/models/smpl")
        smpl_candidates = [
            ("SMPL_NEUTRAL.pkl", os.path.join(local_smpl_dir, "SMPL_NEUTRAL.pkl")),
            ("SMPL_MALE.pkl", os.path.join(local_smpl_dir, "SMPL_MALE.pkl")),
            ("SMPL_FEMALE.pkl", os.path.join(local_smpl_dir, "SMPL_FEMALE.pkl")),
        ]
        for file_name, local_path in smpl_candidates:
            dst_path = os.path.join(smpl_cache_dir, file_name)
            if os.path.exists(dst_path):
                continue
            if os.path.exists(local_path):
                shutil.copyfile(local_path, dst_path)

        # 这些文件同样在 TokenHMR 的 data 目录中提供,直接复制到 PHALP 缓存目录即可.
        local_to_cache = [
            ("smpl_mean_params.npz", os.path.join(local_body_models_dir, "smpl_mean_params.npz")),
            ("SMPL_to_J19.pkl", os.path.join(local_body_models_dir, "SMPL_to_J19.pkl")),
        ]
        for file_name, local_path in local_to_cache:
            dst_path = os.path.join(CACHE_DIR, "phalp/3D", file_name)
            if os.path.exists(dst_path):
                continue
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"缺少本地文件: {local_path}. "
                    "请先运行 `pixi run fetch_demo_data`,确保 `data/body_models` 已下载完成."
                )
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copyfile(local_path, dst_path)

        # ---------------------------------------------------------------------
        # 3) 其余资源使用 UT Austin 镜像下载(替代 Berkeley 站点).
        # ---------------------------------------------------------------------
        base_url = "https://www.cs.utexas.edu/~pavlakos/phalp"
        additional_urls = additional_urls if additional_urls is not None else {}

        download_files = {
            # 3D assets
            "head_faces.npy": [f"{base_url}/3D/head_faces.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "mean_std.npy": [f"{base_url}/3D/mean_std.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            # 下面两个我们会尝试从本地复制,但仍保留下载兜底,防止用户 data 不完整.
            "smpl_mean_params.npz": [f"{base_url}/3D/smpl_mean_params.npz", os.path.join(CACHE_DIR, "phalp/3D")],
            "SMPL_to_J19.pkl": [f"{base_url}/3D/SMPL_to_J19.pkl", os.path.join(CACHE_DIR, "phalp/3D")],
            "texture.npz": [f"{base_url}/3D/texture.npz", os.path.join(CACHE_DIR, "phalp/3D")],
            "bmap_256.npy": [f"{base_url}/bmap_256.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "fmap_256.npy": [f"{base_url}/fmap_256.npy", os.path.join(CACHE_DIR, "phalp/3D")],

            # weights
            "hmar_v2_weights.pth": [f"{base_url}/weights/hmar_v2_weights.pth", os.path.join(CACHE_DIR, "phalp/weights")],
            "pose_predictor.pth": [f"{base_url}/weights/pose_predictor_40006.ckpt", os.path.join(CACHE_DIR, "phalp/weights")],
            "pose_predictor.yaml": [f"{base_url}/weights/config_40006.yaml", os.path.join(CACHE_DIR, "phalp/weights")],

            # AVA dataset (demo_video 不一定用到,但 PHALP 上游默认会准备这些文件)
            "ava_labels.pkl": [f"{base_url}/ava/ava_labels.pkl", os.path.join(CACHE_DIR, "phalp/ava")],
            "ava_class_mapping.pkl": [f"{base_url}/ava/ava_class_mappping.pkl", os.path.join(CACHE_DIR, "phalp/ava")],
        } | additional_urls  # type: ignore

        for file_name, (url, dst_dir) in download_files.items():
            dst_path = os.path.join(dst_dir, file_name)
            if os.path.exists(dst_path):
                continue
            print("Downloading file: " + file_name)
            output = cache_url(url, dst_path)
            if not os.path.exists(dst_path):
                raise FileNotFoundError(f"PHALP 资源下载失败: {file_name}, url={url}, output={output}")

    def setup_hmr(self):
        self.HMAR = TokenHMRPredictor(self.cfg)

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    # ---------------------------------------------------------------------
    # Bench: 用于做吞吐量/FPS 基准测试.
    # - 目标是把"一次性初始化加载成本"(模型/权重加载等)从 FPS 计算里排除掉.
    # - 默认关闭,避免影响正常 demo 输出.
    # ---------------------------------------------------------------------
    bench: bool = False

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:

    """Main function for running the PHALP tracker."""

    bench_enabled = bool(getattr(cfg, "bench", False))

    # -------------------------------------------------------------------------
    # PyTorch 2.6 兼容性:
    # - `torch.load` 的 `weights_only` 默认值从 False 改为 True.
    # - PHALP 的 pose predictor checkpoint(从镜像站点下载)里包含 OmegaConf 的 DictConfig/ListConfig,
    #   会触发 "Unsupported global ... DictConfig" 的反序列化错误.
    # - 这里把 OmegaConf 的配置类型加入 torch 的 allowlist,保持 `weights_only=True` 的安全模式.
    # -------------------------------------------------------------------------
    try:
        import torch
        from omegaconf import DictConfig as OmegaDictConfig
        from omegaconf import ListConfig as OmegaListConfig
        from omegaconf.base import ContainerMetadata

        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([OmegaDictConfig, OmegaListConfig, ContainerMetadata])

        # ---------------------------------------------------------------------
        # 进一步兜底:
        # - 某些 PHALP checkpoint(例如 pose_predictor_40006.ckpt)属于 Lightning 风格,
        #   内部包含较多非 Tensor 对象,在 `weights_only=True` 下容易继续触发 allowlist 问题.
        # - 这里对 PHALP 的 `load_weights` 做 monkeypatch,显式用 `weights_only=False` 加载.
        # - 注意: 仅在你信任 checkpoint 来源时才应该这样做.
        # ---------------------------------------------------------------------
        from phalp.models.hmar.hmar import HMAR
        from phalp.models.predictor.pose_transformer_v2 import Pose_transformer_v2

        if not getattr(Pose_transformer_v2.load_weights, "_tokenhmr_patched", False):
            def _pose_transformer_load_weights(self, path):  # type: ignore[no-redef]
                checkpoint_file = torch.load(path, weights_only=False)
                checkpoint_file_filtered = {
                    k.replace("encoder.", ""): v for k, v in checkpoint_file["state_dict"].items()
                }
                self.encoder.load_state_dict(checkpoint_file_filtered, strict=False)

            _pose_transformer_load_weights._tokenhmr_patched = True  # type: ignore[attr-defined]
            Pose_transformer_v2.load_weights = _pose_transformer_load_weights  # type: ignore[assignment]

        if not getattr(HMAR.load_weights, "_tokenhmr_patched", False):
            def _hmar_load_weights(self, path):  # type: ignore[no-redef]
                checkpoint_file = torch.load(path, weights_only=False)
                state_dict_filt = {}
                for k, v in checkpoint_file["model"].items():
                    if ("encoding_head" in k or "texture_head" in k or "backbone" in k or "smplx_head" in k):
                        state_dict_filt.setdefault(k[5:].replace("smplx", "smpl"), v)
                self.load_state_dict(state_dict_filt, strict=False)

            _hmar_load_weights._tokenhmr_patched = True  # type: ignore[attr-defined]
            HMAR.load_weights = _hmar_load_weights  # type: ignore[assignment]
    except Exception as e:
        # 该兼容逻辑失败时不应直接阻断程序,后续如果仍报反序列化错误再按报错处理.
        log.warning(f"torch safe_globals 初始化失败(可忽略): {e}")

    # -------------------------------------------------------------------------
    # Bench 计时策略:
    # - init_sec: 从开始创建 tracker 到 tracker 构造完成(包含模型/权重加载等一次性成本).
    # - track_sec: 仅统计 `phalp_tracker.track()` 的耗时,用于计算 FPS(排除 init).
    # -------------------------------------------------------------------------
    init_start = time.perf_counter() if bench_enabled else 0.0
    phalp_tracker = PHALP_Prime_TokenHMR(cfg)
    init_end = time.perf_counter() if bench_enabled else 0.0

    track_start = time.perf_counter() if bench_enabled else 0.0
    phalp_tracker.track()
    track_end = time.perf_counter() if bench_enabled else 0.0

    if bench_enabled:
        # ---------------------------------------------------------------------
        # 从输出结果推导帧数:
        # - PHALP 默认会写 `results/{track_dataset}_{video_name}.pkl`
        # - 里面是一个 dict,每个 key 对应一帧,因此 len(dict) 就是处理帧数.
        # ---------------------------------------------------------------------
        frames_processed = 0
        try:
            import joblib
            from hydra.core.hydra_config import HydraConfig

            # `cfg.video.source` 可能是绝对路径 mp4,也可能是图片目录.
            # 这里用 Path.stem/Path.name 兼容两种情况.
            src = Path(str(cfg.video.source))
            video_name = src.stem if src.suffix else src.name
            track_dataset = str(getattr(cfg, "track_dataset", "demo"))

            # Hydra 默认不一定会 chdir 到 output_dir,因此不能依赖 `Path.cwd()`.
            # PHALP 的约定是把结果写入 `cfg.video.output_dir/results/`.
            output_dir = Path(HydraConfig.get().runtime.output_dir)
            results_dir = output_dir / "results"

            expected_pkl_path = results_dir / f"{track_dataset}_{video_name}.pkl"
            pkl_path: Optional[Path] = expected_pkl_path if expected_pkl_path.exists() else None

            if pkl_path is None and results_dir.is_dir():
                # 兜底: 如果文件名不完全匹配(例如 source 是 youtube/pkl),就挑最新的 pkl.
                candidates = list(results_dir.glob("*.pkl"))
                if len(candidates) == 1:
                    pkl_path = candidates[0]
                elif len(candidates) > 1:
                    pkl_path = max(candidates, key=lambda p: p.stat().st_mtime)

            if pkl_path is not None and pkl_path.exists():
                obj = joblib.load(pkl_path)
                if isinstance(obj, dict):
                    frames_processed = len(obj)
        except Exception as e:
            log.warning(f"Bench: 读取结果 pkl 统计帧数失败(将尝试其他方式): {e}")

        if frames_processed <= 0:
            # fallback: 尝试从输入图片目录统计(最适合本次“先拆帧再基准”的场景)
            try:
                src = Path(str(cfg.video.source))
                if src.is_dir():
                    frames_processed = len(list(src.glob("*.jpg")))
            except Exception as e:
                log.warning(f"Bench: 从输入图片目录统计帧数失败: {e}")

        if frames_processed <= 0:
            # fallback: 尝试从拆帧目录统计(仅在 `video.extract_video=true` 时存在)
            try:
                src = Path(str(cfg.video.source))
                video_name = src.stem if src.suffix else src.name
                from hydra.core.hydra_config import HydraConfig
                output_dir = Path(HydraConfig.get().runtime.output_dir)
                img_dir = output_dir / "_DEMO" / video_name / "img"
                if img_dir.is_dir():
                    frames_processed = len(list(img_dir.glob("*.jpg")))
            except Exception as e:
                log.warning(f"Bench: 从拆帧目录统计帧数失败: {e}")

        init_sec = max(0.0, init_end - init_start)
        track_sec = max(0.0, track_end - track_start)
        fps_excluding_init = (frames_processed / track_sec) if (frames_processed > 0 and track_sec > 0) else 0.0
        sec_per_frame = (track_sec / frames_processed) if frames_processed > 0 else 0.0

        # 用 log + print 双输出,避免用户只看 stdout 时错过信息.
        summary = (
            f"BENCH: frames={frames_processed}, "
            f"init_sec={init_sec:.3f}, track_sec={track_sec:.3f}, "
            f"fps_excluding_init={fps_excluding_init:.3f}, sec_per_frame={sec_per_frame:.3f}"
        )
        log.info(summary)
        print(summary)

if __name__ == "__main__":
    main()
