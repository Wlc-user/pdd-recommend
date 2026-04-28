import torch
import torch.nn as nn

class DeepFM(nn.Module):
    """
    DeepFM 精排模型
    ================
    输入：用户特征 + 物品特征 + 上下文特征 → 拼接成一个向量
    输出：点击率预估 (0~1)
    
    结构：
      FM部分（一阶+二阶特征交互）
      + DNN部分（高阶非线性）
      → Sigmoid输出
    """
    def __init__(self, feature_dim, hidden=[256, 128, 64], emb_dim=16):
        super().__init__()
        
        # === FM 一阶部分 ===
        self.linear = nn.Linear(feature_dim, 1)
        
        # === FM 二阶部分 ===
        self.embeddings = nn.ModuleList([
            nn.Embedding(1000, emb_dim) for _ in range(feature_dim)
        ])
        
        # === Deep 部分 ===
        dnn_layers = []
        prev = feature_dim
        for h in hidden:
            dnn_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        self.dnn = nn.Sequential(*dnn_layers)
        self.dnn_out = nn.Linear(hidden[-1], 1)
        
        # === 最终输出 ===
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # FM 一阶
        fm_first = self.linear(x)  # [B, 1]
        
        # FM 二阶（简化：用内积近似特征交互）
        fm_second = 0.5 * torch.sum(
            torch.pow(torch.sum(x.unsqueeze(-1) * x.unsqueeze(1), dim=1), 2) -
            torch.sum(torch.pow(x, 2), dim=1, keepdim=True),
            dim=1, keepdim=True
        )  # [B, 1]
        
        # Deep 部分
        dnn = self.dnn(x)
        dnn = self.dnn_out(dnn)  # [B, 1]
        
        # 组合输出
        out = fm_first + fm_second + dnn
        return self.sigmoid(out)