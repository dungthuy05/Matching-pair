import numpy as np
import torch
import torch.nn as nn


class SimpleMultiHeadAttention(nn.Module):
    """
    Multi-head attention tự viết để dễ lấy Q, K, V & attention map.
    """

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        assert d_model % nhead == 0, "d_model phải chia hết cho nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead  # dim mỗi head

        # Linear để sinh Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Linear tổng hợp lại sau khi concat các head
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, need_attn: bool = False):
        """
        q: (B, Nq, d_model)
        k: (B, Nk, d_model)
        v: (B, Nk, d_model)

        return:
          out: (B, Nq, d_model)
          nếu need_attn=True:
            attn: (B, nhead, Nq, Nk)
            Q, K, V: (B, nhead, Nq/Nk, d_head)
        """
        B, Nq, _ = q.shape
        _, Nk, _ = k.shape

        # 1) Linear projections
        Q = self.W_q(q)  # (B, Nq, d_model)
        K = self.W_k(k)  # (B, Nk, d_model)
        V = self.W_v(v)  # (B, Nk, d_model)

        # 2) Tách thành nhiều head -> (B, nhead, Nq/Nk, d_head)
        Q = Q.view(B, Nq, self.nhead, self.d_head).transpose(1, 2)
        K = K.view(B, Nk, self.nhead, self.d_head).transpose(1, 2)
        V = V.view(B, Nk, self.nhead, self.d_head).transpose(1, 2)

        # 3) Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head**0.5)
        attn = scores.softmax(dim=-1)  # (B, nhead, Nq, Nk)

        # 4) Weighted sum
        context = torch.matmul(attn, V)  # (B, nhead, Nq, d_head)

        # 5) Ghép head lại
        context = context.transpose(1, 2).contiguous().view(B, Nq, self.d_model)

        out = self.W_o(context)  # (B, Nq, d_model)

        if need_attn:
            return out, attn, Q, K, V
        else:
            return out


class SelfAttentionBlock(nn.Module):
    """
    Block self-attention đơn giản dùng nn.TransformerEncoder.
    """

    def __init__(self, d_model: int, nhead: int = 4, ff_ratio: int = 4):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_ratio * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, d_model)
        return: (N, d_model)
        """
        x = x.unsqueeze(0)  # (1, N, d_model)
        x = self.encoder(x)
        x = x.squeeze(0)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention giữa hai ảnh, dùng SimpleMultiHeadAttention.
    """

    def __init__(self, d_model: int, nhead: int = 4, ff_ratio: int = 4):
        super().__init__()
        self.mha = SimpleMultiHeadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_ratio * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(ff_ratio * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        need_details: bool = False,
    ):
        """
        q:  (Nq, d_model)
        kv: (Nk, d_model)

        return:
          out: (Nq, d_model)
          attn_weights: (Nq, Nk)  # avg over heads
          nếu need_details=True:
            full_attn: (1, nhead, Nq, Nk)
            Q, K, V:   (1, nhead, Nq/Nk, d_head)
        """
        q_b = q.unsqueeze(0)  # (1, Nq, d_model)
        kv_b = kv.unsqueeze(0)  # (1, Nk, d_model)

        if need_details:
            attn_out, attn, Q, K, V = self.mha(q_b, kv_b, kv_b, need_attn=True)
        else:
            attn_out = self.mha(q_b, kv_b, kv_b, need_attn=False)
            attn = None
            Q = K = V = None

        x = self.norm1(q_b + attn_out)
        x2 = self.ffn(x)
        x = self.norm2(x + x2)

        x = x.squeeze(0)  # (Nq, d_model)

        attn_weights = None
        if need_details and attn is not None:
            # attn: (1, nhead, Nq, Nk)
            attn_weights = attn.mean(dim=1).squeeze(0)  # (Nq, Nk)
            return x, attn_weights, attn, Q, K, V

        return x, attn_weights, None, None, None


class TransformerMatcher(nn.Module):
    """
    Transformer-based matcher cho keypoint descriptors.
    """

    def __init__(
        self,
        desc_dim: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        ff_ratio: int = 4,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        # projector descriptor 128 -> d_model
        self.desc_proj = nn.Linear(desc_dim, d_model)

        # projector vị trí (x,y) -> d_model
        self.pos_proj = nn.Linear(2, d_model)

        # self-attention cho từng ảnh
        self.self_attn1 = SelfAttentionBlock(d_model, nhead, ff_ratio)
        self.self_attn2 = SelfAttentionBlock(d_model, nhead, ff_ratio)

        # cross-attention hai chiều
        self.cross_attn_12 = CrossAttentionBlock(d_model, nhead, ff_ratio)
        self.cross_attn_21 = CrossAttentionBlock(d_model, nhead, ff_ratio)

        self.to(self.device)

    def encode_points(self, pts, desc):
        """
        pts:  (N, 2)  numpy/torch
        desc: (N, D)
        return: (N, d_model)
        """
        if not torch.is_tensor(pts):
            pts = torch.from_numpy(pts)
        if not torch.is_tensor(desc):
            desc = torch.from_numpy(desc)

        pts = pts.float().to(self.device)
        desc = desc.float().to(self.device)

        f_desc = self.desc_proj(desc)  # (N, d_model)
        f_pos = self.pos_proj(pts)  # (N, d_model)
        f = f_desc + f_pos  # (N, d_model)
        return f

    @torch.no_grad()
    def forward(
        self,
        pts1,
        desc1,
        pts2,
        desc2,
        mutual_check: bool = True,
        min_score: float = 0.0,
    ):
        """
        Trả về list match (idx1, idx2, score) dưới dạng numpy array (M,3).
        Dùng cho inference: không trả Q,K,V.
        """
        if desc1 is None or desc2 is None:
            return np.zeros((0, 3), dtype=np.float32)

        if len(desc1) == 0 or len(desc2) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # 1) Encode descriptor + position
        f1 = self.encode_points(pts1, desc1)  # (N1, d_model)
        f2 = self.encode_points(pts2, desc2)  # (N2, d_model)

        # 2) Self-attention từng ảnh
        f1_ctx = self.self_attn1(f1)  # (N1, d_model)
        f2_ctx = self.self_attn2(f2)  # (N2, d_model)

        # 3) Cross-attention: ảnh1 nhìn ảnh2 và ngược lại
        _, attn12, _, _, _, _ = self.cross_attn_12(f1_ctx, f2_ctx, need_details=True)
        _, attn21, _, _, _, _ = self.cross_attn_21(f2_ctx, f1_ctx, need_details=True)

        # 4) Mutual best matching + threshold
        i_to_j = attn12.argmax(dim=1)  # (N1,)
        j_to_i = attn21.argmax(dim=1)  # (N2,)

        scores_12 = torch.gather(attn12, 1, i_to_j.unsqueeze(1)).squeeze(1)  # (N1,)

        matches = []
        for i in range(len(i_to_j)):
            j = i_to_j[i].item()
            if mutual_check:
                if j >= len(j_to_i):
                    continue
                if j_to_i[j].item() != i:
                    continue

            score = scores_12[i].item()
            if score < min_score:
                continue

            matches.append((i, j, score))

        if len(matches) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        matches = np.array(matches, dtype=np.float32)
        return matches

    @torch.no_grad()
    def forward_with_details(
        self,
        pts1,
        desc1,
        pts2,
        desc2,
    ):
        """
        Giống forward nhưng trả thêm:
          - attn12: (N1, N2) attention (avg over heads)
          - full_attn12: (1, nhead, N1, N2)
        Dùng cho visualization.
        """
        if desc1 is None or desc2 is None:
            return (
                np.zeros((0, 3), dtype=np.float32),
                None,
                None,
            )

        if len(desc1) == 0 or len(desc2) == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                None,
                None,
            )

        f1 = self.encode_points(pts1, desc1)
        f2 = self.encode_points(pts2, desc2)

        f1_ctx = self.self_attn1(f1)
        f2_ctx = self.self_attn2(f2)

        _, attn12, full_attn12, Q12, K12, V12 = self.cross_attn_12(
            f1_ctx, f2_ctx, need_details=True
        )
        _, attn21, full_attn21, Q21, K21, V21 = self.cross_attn_21(
            f2_ctx, f1_ctx, need_details=True
        )

        # Tạo matches giống forward()
        i_to_j = attn12.argmax(dim=1)
        j_to_i = attn21.argmax(dim=1)
        scores_12 = torch.gather(attn12, 1, i_to_j.unsqueeze(1)).squeeze(1)

        matches = []
        for i in range(len(i_to_j)):
            j = i_to_j[i].item()
            if j >= len(j_to_i):
                continue
            if j_to_i[j].item() != i:
                continue

            score = scores_12[i].item()
            matches.append((i, j, score))

        if len(matches) == 0:
            matches_np = np.zeros((0, 3), dtype=np.float32)
        else:
            matches_np = np.array(matches, dtype=np.float32)

        return matches_np, attn12.cpu().numpy(), full_attn12.cpu().numpy()
