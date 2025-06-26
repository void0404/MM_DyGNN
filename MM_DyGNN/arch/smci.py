import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttentionLayerTopK(nn.Module):
    """
    CrossAttentionLayer 与 Top-K_Sparse_Attention
    """

    def __init__(
            self,
            model_input_dim,  # 输入维度 c
            model_dim,  # 注意力投影维度
            feed_forward_dim=1024,
            num_heads=8,
            dropout=0.,
            mask=False
    ):
        super().__init__()

        self.model_input_dim = model_input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.mask = mask

        # 映射到 Q_1, Q_2, K_1, K_2, V
        self.FC_Q_1 = nn.Linear(model_input_dim, model_dim)
        self.FC_Q_2 = nn.Linear(model_input_dim, model_dim)
        self.FC_K_1 = nn.Linear(model_input_dim, model_dim)
        self.FC_K_2 = nn.Linear(model_input_dim, model_dim)
        self.FC_V = nn.Linear(model_input_dim, model_dim)

        # 多头注意力输出再映射回原输入维度
        self.out_proj = nn.Linear(model_dim, model_input_dim)

        # Feed-Forward 子层
        self.feed_forward = nn.Sequential(
            nn.Linear(model_input_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_input_dim),
        )

        # 残差 & 归一化 & dropout
        self.ln1 = nn.LayerNorm(model_input_dim)
        self.ln2 = nn.LayerNorm(model_input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 在 Top-K 稀疏注意力中，需要用到的温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 只保留 2 个可学习权重，分别对应 N_k/8 和 N_k/4
        self.attn_alphas = nn.Parameter(torch.ones(2) / 2, requires_grad=True)

    def forward(self, x, x_au1, x_au2):
        """
        x      : [bs, c, N, L]
        x_au1  : [bs, c, N, L] (务必保持与 x 同 shape)
        x_au2  : [bs, c, N, L] (务必保持与 x 同 shape)

        输出    : [bs, c, N, L]
        """

        # ========== 1) 确保输入形状一致 ========== #
        assert x.shape == x_au1.shape, f"x and x_au1 shape mismatch: {x.shape} vs {x_au1.shape}"
        assert x.shape == x_au2.shape, f"x and x_au2 shape mismatch: {x.shape} vs {x_au2.shape}"

        # ========== 2) 转置: [bs, c, N, L] -> [bs, L, N, c] ========== #
        x = x.transpose(1, 3)
        x_au1 = x_au1.transpose(1, 3)
        x_au2 = x_au2.transpose(1, 3)
        # 现在 x 形状: (bs, L, N, c)
        bs, L, N, c = x.shape

        # ========== 3) 分别线性映射到 Q_1, Q_2, K_1, K_2, V ========== #
        Q_1 = self.FC_Q_1(x_au1)  # [bs, L, N, model_dim]
        Q_2 = self.FC_Q_2(x_au2)  # [bs, L, N, model_dim]
        K_1 = self.FC_K_1(x)      # [bs, L, N, model_dim]
        K_2 = self.FC_K_2(x)      # [bs, L, N, model_dim]
        V = self.FC_V(x)          # [bs, L, N, model_dim]

        # ========== 4) 把多头拆分: -> [bs*num_heads, L, N, head_dim] ========== #
        def split_heads(tensor):
            # [bs, L, N, model_dim] -> [bs, L, N, num_heads, head_dim]
            tensor = tensor.view(bs, L, N, self.num_heads, self.head_dim)
            # -> [bs, num_heads, L, N, head_dim]
            tensor = tensor.permute(0, 3, 1, 2, 4)
            # 合并 bs 和 num_heads -> [bs*num_heads, L, N, head_dim]
            tensor = tensor.reshape(bs * self.num_heads, L, N, self.head_dim)
            return tensor

        q_1 = split_heads(Q_1)
        q_2 = split_heads(Q_2)
        k_1 = split_heads(K_1)
        k_2 = split_heads(K_2)
        v = split_heads(V)

        # ========== 5) 做归一化再计算注意力分数 ========== #
        # q,k 先归一化，以匹配 "Top-K_Sparse_Attention" 中的做法
        q_1 = F.normalize(q_1, dim=-1)
        q_2 = F.normalize(q_2, dim=-1)
        k_1 = F.normalize(k_1, dim=-1)
        k_2 = F.normalize(k_2, dim=-1)

        # 让 temperature 变成 [bs*num_heads, 1, 1, 1]
        temp = self.temperature.repeat(bs, 1, 1).reshape(bs * self.num_heads, 1, 1, 1)

        # 形状: [bs*num_heads, L, N, head_dim] x [bs*num_heads, L, head_dim, N]
        attn1 = torch.matmul(q_1, k_1.transpose(-2, -1))  # => [bs*num_heads, L, N, N]
        attn2 = torch.matmul(q_2, k_2.transpose(-2, -1))  # => [bs*num_heads, L, N, N]

        attn = (attn1 + attn2) * temp  # => [bs*num_heads, L, N, N]

        # ========== 6) 若需要mask，这里演示lower-triangular mask ========== #
        if self.mask:
            # mask shape: [N, N]
            mask = torch.ones(N, N, dtype=torch.bool, device=attn.device).tril()
            attn = attn.masked_fill(~mask, float('-inf'))

        # ========== 7) Top-K 稀疏注意力：仅使用 2 种截断范围 (N_k/8 与 N_k/4) + softmax + 加权组合 ========== #
        _, _, N_q, N_k = attn.shape  # N_q = N_k = N

        # 建立 2 个零mask
        mask1 = torch.zeros_like(attn, requires_grad=False)
        mask2 = torch.zeros_like(attn, requires_grad=False)

        # a) top_k = N_k / 8
        index1 = torch.topk(attn, k=max(1, int(N_k / 2)), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index1, 1.)
        attn_top1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        # b) top_k = N_k / 4
        index2 = torch.topk(attn, k=max(1, int(N_k / 4)), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index2, 1.)
        attn_top2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        # 进行 softmax
        attn_top1 = F.softmax(attn_top1, dim=-1)
        attn_top2 = F.softmax(attn_top2, dim=-1)

        # 不同截断范围对应不同输出
        out1 = torch.matmul(attn_top1, v)
        out2 = torch.matmul(attn_top2, v)

        # 使用 softmax 确保权重和为 1
        alphas = F.softmax(self.attn_alphas, dim=0)

        # 加权融合
        out = out1 * alphas[0] + out2 * alphas[1]

        # ========== 8) 合并多头, 再映射回到 model_input_dim ========== #
        # out shape: [bs*num_heads, L, N, head_dim]
        out = out.reshape(bs, self.num_heads, L, N, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4)  # => [bs, L, N, num_heads, head_dim]
        out = out.reshape(bs, L, N, self.model_dim)
        out = self.out_proj(out)  # => [bs, L, N, model_input_dim]

        # ========== 9) 残差 + LN + 前馈 ========== #
        # 第一层: 注意力输出 + 残差 + LN
        residual = x  # [bs, L, N, c]
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        # 第二层: FFN + 残差 + LN
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        # ========== 10) 还原回 [bs, c, N, L] 形状 ========== #
        out = out.transpose(1, 3)  # [bs, L, N, c] -> [bs, c, N, L]
        return out,attn,attn_top2


# ============== 测试代码 ============== #
if __name__ == "__main__":
    # 假设输入: [bs, c, N, L]
    bs, c, N, L = 2, 16, 8, 32

    # 输入保持一致
    x = torch.randn(bs, c, N, L)
    x_au1 = torch.randn(bs, c, N, L)
    x_au2 = torch.randn(bs, c, N, L)

    model = CrossAttentionLayerTopK(
        model_input_dim=c,
        model_dim=32,
        feed_forward_dim=64,
        num_heads=4,
        dropout=0.1,
        mask=False
    )

    out = model(x, x_au1, x_au2)
    print("输入 x, x_au1, x_au2 的形状：", x.shape)
    print("输出 out 的形状：", out.shape)
    # out 应当是 [2, 16, 8, 32]