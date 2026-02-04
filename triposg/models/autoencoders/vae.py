from typing import Optional, Tuple

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor


class DiagonalGaussianDistribution(object):
    """對角高斯分佈類別 - 用於VAE的潛在空間表示"""

    def __init__(
        self,
        parameters: torch.Tensor,  # 包含均值和對數變異數的參數張量
        deterministic: bool = False,  # 是否使用確定性模式（不添加隨機雜訊）
        feature_dim: int = 1,  # 特徵維度，用於分割均值和變異數
    ):
        self.parameters = parameters
        self.feature_dim = feature_dim
        # 將參數張量沿指定維度分成兩部分：均值和對數變異數
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=feature_dim)
        # 限制對數變異數的範圍，防止數值不穩定
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        # 計算標準差：std = exp(0.5 * logvar)
        self.std = torch.exp(0.5 * self.logvar)
        # 計算變異數：var = exp(logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            # 確定性模式下，標準差和變異數都設為0（無隨機性）
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """從高斯分佈中採樣"""
        # 確保採樣的張量與參數在同一設備上且資料類型相同
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        # 重參數化技巧：x = μ + σ * ε，其中ε~N(0,1)
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        """計算KL散度（Kullback-Leibler divergence）"""
        if self.deterministic:
            # 確定性情況下KL散度為0
            return torch.Tensor([0.0])
        else:
            if other is None:
                # 與標準常態分佈N(0,1)的KL散度
                # KL(N(μ,σ²)||N(0,1)) = 0.5 * (μ² + σ² - 1 - log(σ²))
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                # 與另一個高斯分佈的KL散度
                # KL(N(μ₁,σ₁²)||N(μ₂,σ₂²)) = 0.5 * ((μ₁-μ₂)²/σ₂² + σ₁²/σ₂² - 1 - log(σ₁²) + log(σ₂²))
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(
        self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]
    ) -> torch.Tensor:
        """計算負對數似然（Negative Log-Likelihood）"""
        if self.deterministic:
            # 確定性情況下負對數似然為0
            return torch.Tensor([0.0])
        # log(2π)
        logtwopi = np.log(2.0 * np.pi)
        # NLL = 0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        """傳回分佈的眾數（對高斯分佈來說就是均值）"""
        return self.mean
