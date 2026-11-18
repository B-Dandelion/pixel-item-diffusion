"""학습이 완료된 DDPM 모델을 불러와 이미지를 생성 및 저장"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.utils import build_models, load_checkpoint, save_images, seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    #실행 옵션 설정하기
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/samples/generated")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("count must be positive")
    started = time.perf_counter()
    seed_everything(args.seed)

    # 체크포인트에서 모델 설정 복원
    metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = metadata["config"]

    model, diffusion = build_models(config, torch.device(args.device))
    load_checkpoint(args.checkpoint, model)

    ## 이미지 생성 ##
    images = diffusion.sample(model, args.count, config["image_size"], config["in_channels"], args.seed)
    
    if not torch.isfinite(images).all():
        raise RuntimeError("Sampling produced non-finite pixels")

    # 생성된 이미지 저장
    save_images(images, args.output)
    report = dict(checkpoint=str(args.checkpoint), seed=args.seed, count=args.count,
                  diffusion_steps=diffusion.steps, global_step=metadata["progress"]["global_step"],
                  smoke_only=config.get("smoke_only", False),
                  elapsed_seconds=round(time.perf_counter() - started, 3))
    (Path(args.output) / "sampling.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
