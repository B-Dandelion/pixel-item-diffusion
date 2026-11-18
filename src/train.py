"""
11.18일 기준 미완성 상태, 학습도 돌리지 않음
DDPM의 학습 목표는 노이즈가 추가된 이미지에서 해당 노이즈를 예측하도록 U-Net을 학습시키는 것.
DDPM 학습, 검증, 체크포인트 저장 및 학습 재개를 수행
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import PixelItemDataset
from src.utils import (build_models, file_hash, load_checkpoint, load_config,
                       save_checkpoint, save_images, seed_everything)


def make_loader(dataset, config, epoch, next_batch=0, shuffle=True):
    # 학습 데이터 공급
    generator = torch.Generator().manual_seed(config["seed"] + epoch)

    indices = torch.randperm(len(dataset), generator=generator).tolist() if shuffle else list(range(len(dataset)))
    indices = indices[next_batch * config["batch_size_initial"]:]
    # next_batch는 학습 도중 중단될 경우를 대비

    return DataLoader(Subset(dataset, indices), batch_size=config["batch_size_initial"],
                      num_workers=config["num_workers"], pin_memory=torch.cuda.is_available(),
                      generator=generator)


@torch.no_grad() #검증과정이므로 gradient 계산 ㄴㄴ
def evaluate(model, diffusion, dataset, config, device):
    #모델 검증
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    total, count = 0.0, 0

    for batch in make_loader(dataset, config, 0, shuffle=False):
        batch = batch.to(device, non_blocking=True)

        loss = diffusion.training_loss(model, batch, generator)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        total += loss.item() * len(batch)
        count += len(batch)
    model.train()
    return total / count


def write_history(run_dir, history):
    # 학습 기록
    with (run_dir / "loss.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "global_step", "train_loss", "val_loss", "train_seconds"])
        writer.writeheader()
        writer.writerows(history)


def main(argv=None):
    #전체 학습 내용
    parser = argparse.ArgumentParser(description=__doc__)

    #실행 옵션
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--run-dir", default="checkpoints/ddpm_pixel_items_baseline")
    parser.add_argument("--resume", help="Resume model, optimizer, RNG, epoch and batch cursor")
    parser.add_argument("--epochs", type=int, help="Total target epochs, including completed epochs")
    parser.add_argument("--max-steps", type=int, help="Stop after this many optimizer steps in this invocation")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args(argv)
    started = time.perf_counter()
    config = load_config(args.config)
    epochs = args.epochs if args.epochs is not None else config["epochs_initial"]
    if epochs < 1 or config["batch_size_initial"] < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("epochs, batch size and max steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; fix the GPU environment or explicitly use --device cpu.")
    seed_everything(config["seed"])
    device = torch.device(args.device)
    train_data = PixelItemDataset(config["manifest"], "train", config["image_size"])
    val_data = PixelItemDataset(config["manifest"], "val", config["image_size"])
    manifest_hash = file_hash(config["manifest"])
    run_dir = Path(args.run_dir)
    latest = run_dir / "latest.pt"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise SystemExit("Run directory is not empty. Use --resume or a new --run-dir.")
    if args.resume and latest.exists() and latest.resolve() != Path(args.resume).resolve():
        raise SystemExit("Resume an older checkpoint into a new --run-dir to preserve current results.")
    
    # 모델 및 Optimizer 생성
    model, diffusion = build_models(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate_initial"])
    progress = dict(epoch=0, next_batch=0, global_step=0, epoch_loss_sum=0.0,
                    epoch_examples=0, epoch_train_seconds=0.0, history=[])
    if args.resume:
        metadata = torch.load(args.resume, map_location="cpu", weights_only=True)
        if metadata["config"] != config or metadata["manifest_sha256"] != manifest_hash:
            raise SystemExit("Resume config or dataset manifest differs from the checkpoint.")
        checkpoint = load_checkpoint(args.resume, model, optimizer, restore_random=True)
        progress = checkpoint["progress"]
        del metadata, checkpoint
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_history(run_dir, progress["history"])
    total_batches = math.ceil(len(train_data) / config["batch_size_initial"])
    invocation_steps = 0
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"train/val: {len(train_data)}/{len(val_data)}", flush=True)

    # 학습 반복문
    while progress["epoch"] < epochs:
        epoch = progress["epoch"]
        model.train()
        loader = make_loader(train_data, config, epoch, progress["next_batch"])
        for batch in loader:
            step_started = time.perf_counter()
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # training_loss => 이미지에 더해진 실제 노이즈와 모델이 예측한 노이즈의 차이를 계산한 값
            # backward => 손실을 줄이기 위한 기울기 계산
            loss = diffusion.training_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; checkpoint was not advanced")
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            progress["epoch_loss_sum"] += loss_value * len(batch)
            progress["epoch_examples"] += len(batch)
            progress["epoch_train_seconds"] += time.perf_counter() - step_started
            progress["next_batch"] += 1
            progress["global_step"] += 1
            invocation_steps += 1
            finished_epoch = progress["next_batch"] == total_batches
            stopped = args.max_steps is not None and invocation_steps >= args.max_steps
            if finished_epoch:
                record = dict(epoch=epoch + 1, global_step=progress["global_step"],
                              train_loss=progress["epoch_loss_sum"] / progress["epoch_examples"],
                              val_loss=evaluate(model, diffusion, val_data, config, device),
                              train_seconds=progress["epoch_train_seconds"])
                progress["history"].append(record)
                progress.update(epoch=epoch + 1, next_batch=0, epoch_loss_sum=0.0,
                                epoch_examples=0, epoch_train_seconds=0.0)
                write_history(run_dir, progress["history"])
                print(json.dumps(record), flush=True)

                # 학습 중 이미지 생성(과제 요건 충족 목적)
                sample_every = config["sample_every_epochs"]
                if sample_every > 0 and (epoch + 1) % sample_every == 0:
                    images = diffusion.sample(model, config["sample_count"], seed=config["fixed_sample_seed"])
                    save_images(images, run_dir / "samples" / f"epoch_{epoch + 1:04d}")
            if finished_epoch or stopped:
                # 유연한 학습 중단과 재개를 위한 체크포인트 설정
                save_checkpoint(latest, model, optimizer, config, progress, manifest_hash)
                checkpoint_every = config["checkpoint_every_epochs"]
                if finished_epoch and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                    save_checkpoint(run_dir / f"epoch_{epoch + 1:04d}.pt", model, optimizer, config, progress, manifest_hash)
            if stopped:
                break
        if args.max_steps is not None and invocation_steps >= args.max_steps:
            break
    summary = dict(completed_epochs=progress["epoch"], next_batch=progress["next_batch"],
                   global_step=progress["global_step"], invocation_steps=invocation_steps,
                   elapsed_seconds=round(time.perf_counter() - started, 3), device=str(device),
                   peak_vram_mib=round(torch.cuda.max_memory_allocated() / 1024**2, 2) if device.type == "cuda" else 0,
                   checkpoint=latest.as_posix())
    (run_dir / "last_invocation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
