"""Measure a synthetic forward/backward pass without an optimizer or training data."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import build_models, load_config, seed_everything


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this memory probe")
    config = load_config("configs/base.json")
    seed_everything(config["seed"])
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, diffusion = build_models(config, device)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    # Probe uses arbitrary tensors, not any collected images; weights are never updated.
    x = torch.zeros(config["batch_size_initial"], config["in_channels"], 32, 32, device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = diffusion.training_loss(model, x)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
    parameters = sum(p.numel() for p in model.parameters())
    result = dict(purpose="synthetic_forward_backward_only_no_optimizer_no_checkpoint",
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), batch=config["batch_size_initial"],
                  dtype="float32", parameters=parameters, elapsed_seconds=round(elapsed, 3),
                  peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2, 2),
                  peak_reserved_mib=round(torch.cuda.max_memory_reserved()/1024**2, 2),
                  adam_parameter_gradient_moments_estimate_mib=round(parameters*16/1024**2, 2),
                  finite_gradients=True, unchanged_parameters=True,
                  note="Peak includes parameter clones for verification; excludes Adam state, data IO and sampling.")
    output = Path("results/model_probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
