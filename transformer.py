import torch
import einops
import math

class CausalTransformer(torch.nn.Module):
    def __init__(self, n_heads: int, d_k: int, d_v: int, d_in: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        scale = math.sqrt(d_in)

        self.Wq = torch.nn.Parameter(torch.randn(d_in, n_heads * d_k) / scale)
        self.Wk = torch.nn.Parameter(torch.randn(d_in, n_heads * d_k) / scale)
        self.Wv = torch.nn.Parameter(torch.randn(d_in, n_heads * d_v) / scale)

        self.Wo = torch.nn.Parameter(torch.randn(n_heads * d_v, d_in) / scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_in); qkv: (B, N, d_k/v)
        q = einops.einsum(x, self.Wq, "B N d_in, d_in d_k -> B N d_k")
        k = einops.einsum(x, self.Wk, "B N d_in, d_in d_k -> B N d_k")
        v = einops.einsum(x, self.Wv, "B N d_in, d_in d_v -> B N d_v")

        head_q = einops.rearrange(q, "B N (H d_k) -> B H N d_k", H=self.n_heads)
        head_k = einops.rearrange(k, "B N (H d_k) -> B H N d_k", H=self.n_heads)
        head_v = einops.rearrange(v, "B N (H d_v) -> B H N d_v", H=self.n_heads)

        raw_scores = einops.einsum(head_q, head_k, "B H N_q d_k, B H N_k d_k -> B H N_q N_k")

        mask = torch.triu(torch.ones_like(raw_scores), diagonal=1).bool()
        masked_scores = raw_scores.masked_fill(mask, -torch.inf)

        scores = torch.softmax(masked_scores / math.sqrt(self.d_k), dim=-1)

        head_attn = einops.einsum(scores, head_v, "B H N_q N_k, B H N_k d_v -> B H N_q d_v")
        attn = einops.rearrange(head_attn, "B H N_q d_v -> B N_q (H d_v)")

        out = einops.einsum(attn, self.Wo, "B N_q d_v, d_v d_in -> B N_q d_in")

        return out

def main():
    batch_size, n_tokens, d_in = 32, 64, 128
    x = torch.rand(batch_size, n_tokens, d_in)
    transformer = CausalTransformer(n_heads=2, d_k=512, d_v=1028, d_in=d_in)
    return transformer(x)

if __name__ == "__main__":
    out = main()
    print(out.shape)
