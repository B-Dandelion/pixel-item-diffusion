"""DDPM 조상 샘플링 구현"""

import torch
from torch import nn
from torch.nn import functional as F


class DDPM(nn.Module):
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        if steps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("Need steps >= 2 and 0 < beta_start < beta_end < 1")
        self.steps = steps
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
        alphas = 1 - betas
        alpha_bars = alphas.cumprod(0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bars[:-1]))
        for name, value in dict(
            betas=betas, alpha_bars=alpha_bars,
            sqrt_alpha_bars=alpha_bars.sqrt(), sqrt_one_minus_alpha_bars=(1 - alpha_bars).sqrt(),
            #사후분포의 분산 계산
            posterior_variance=betas * (1 - previous) / (1 - alpha_bars),
            posterior_x0_coef=betas * previous.sqrt() / (1 - alpha_bars),
            posterior_xt_coef=(1 - previous) * alphas.sqrt() / (1 - alpha_bars),
        ).items():
            self.register_buffer(name, value.float())

    @staticmethod
    def extract(values, timesteps):
        return values[timesteps][:, None, None, None]

    def q_sample(self, clean, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean)
        return (self.extract(self.sqrt_alpha_bars, timesteps) * clean
                + self.extract(self.sqrt_one_minus_alpha_bars, timesteps) * noise)

    def training_loss(self, model, clean, generator=None):
        t = torch.randint(self.steps, (clean.shape[0],), device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        return F.mse_loss(model(self.q_sample(clean, t, noise), t), noise)

    @torch.no_grad()
    def p_sample(self, model, noisy, timesteps, generator=None):
        predicted_noise = model(noisy, timesteps)
        clean = ((noisy - self.extract(self.sqrt_one_minus_alpha_bars, timesteps) * predicted_noise)
                 / self.extract(self.sqrt_alpha_bars, timesteps)).clamp(-1, 1)
        #사후분포를 이용한 역확산
        mean = (self.extract(self.posterior_x0_coef, timesteps) * clean
                + self.extract(self.posterior_xt_coef, timesteps) * noisy)
        noise = torch.randn(noisy.shape, device=noisy.device, dtype=noisy.dtype, generator=generator)
        nonzero = (timesteps > 0).to(noisy.dtype)[:, None, None, None]
        return mean + nonzero * self.extract(self.posterior_variance, timesteps).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, count=16, image_size=32, channels=3, seed=42):
        if count < 1:
            raise ValueError("count must be positive")
        device = self.betas.device
        generator = torch.Generator(device=device).manual_seed(seed)
        noisy = torch.randn((count, channels, image_size, image_size), device=device, generator=generator)
        was_training = model.training
        model.eval()
        try:
            # 1000단계부터 0단계까지 순차적으로 역확산 수행
            for step in reversed(range(self.steps)):
                t = torch.full((count,), step, device=device, dtype=torch.long)
                noisy = self.p_sample(model, noisy, t, generator)
        finally:
            model.train(was_training)
        return noisy.clamp(-1, 1)
