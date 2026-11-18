"""재현 설정, 모델 초기화, 체크포인트 관리 및 이미지 저장"""

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.diffusion import DDPM
from src.unet import UNet


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["image_size"] != 32 or config["in_channels"] != 3:
        raise ValueError("This baseline supports 32x32 RGB only")
    return config


def build_models(config, device):
    model = UNet(config["in_channels"], config["base_channels"], config["channel_multipliers"]).to(device)
    diffusion = DDPM(config["diffusion_steps"], config["beta_start"], config["beta_end"]).to(device)
    return model, diffusion


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_state():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]])


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))


def save_checkpoint(path, model, optimizer, config, progress, manifest_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(version=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    config=config, progress=dict(progress), manifest_sha256=manifest_hash,
                    rng=rng_state()), temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer=None, restore_random=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_random:
        restore_rng(checkpoint["rng"])
    return checkpoint


def save_images(images, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = images.detach().cpu().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(0, 2, 3, 1).numpy()
    columns = math.ceil(math.sqrt(len(arrays)))
    grid = Image.new("RGB", (columns * 32, math.ceil(len(arrays) / columns) * 32), (24, 24, 24))
    for index, array in enumerate(arrays):
        im = Image.fromarray(array)
        im.save(output_dir / f"sample_{index:03d}.png")
        grid.paste(im, ((index % columns) * 32, (index // columns) * 32))
    grid.save(output_dir / "grid.png")
    grid.resize((grid.width * 4, grid.height * 4), Image.Resampling.NEAREST).save(output_dir / "grid_preview.png")
    return output_dir / "grid.png"
