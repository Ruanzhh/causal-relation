import torch
import torch.nn as nn

# 定义两个正态分布的参数
mu1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
sigma1 = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float32)

mu2 = torch.tensor([0.0, 0.0], dtype=torch.float32)
sigma2 = torch.tensor([[1.0, 0.0], [0.0, 3.0]], dtype=torch.float32)

# 创建两个多维正态分布对象
dist1 = torch.distributions.MultivariateNormal(mu1, sigma1)
dist2 = torch.distributions.MultivariateNormal(mu2, sigma2)

# 生成一些样本
samples = dist1.sample((1000,))  # 从第一个分布中采样一些样本
print(samples)
# 计算两个分布的概率密度函数（PDF）
pdf1 = torch.exp(dist1.log_prob(samples))
pdf2 = torch.exp(dist2.log_prob(samples))
print(pdf1.sum())
print(pdf2.sum())
# 创建nn.KLDivLoss损失函数
criterion = nn.KLDivLoss(reduction='batchmean')

# 计算KL散度
kl_divergence = criterion(torch.log(pdf1), pdf2)

print("KL散度值:", kl_divergence.item())

