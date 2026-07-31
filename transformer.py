import torch
import einops
import math

class CausalTransformer(torch.nn.Module):
    def __init__(self, d_k: int, d_v: int, d_in: int) -> None:
        super().__init__()
        scale = math.sqrt(d_in)
        self.Wq = torch.nn.Parameter(torch.randn(d_in, d_k) / scale)
        self.Wk = torch.nn.Parameter(torch.randn(d_in, d_k) / scale)
        self.Wv = torch.nn.Parameter(torch.randn(d_in, d_v) / scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_in); qkv: (B, N, d_k/v)
        q = einops.einsum(x, self.Wq, "B N d_in, d_in d_k -> B N d_k")
        k = einops.einsum(x, self.Wk, "B N d_in, d_in d_k -> B N d_k")
        v = einops.einsum(x, self.Wv, "B N d_in, d_in d_v -> B N d_v")

        raw_scores = einops.einsum(q, k, "B N_q d_k, B N_k d_k -> B N_q N_k")

        mask = torch.triu(torch.ones_like(raw_scores), diagonal=1).bool()
        masked_scores = raw_scores.masked_fill(mask, -torch.inf)

        scores = torch.softmax(masked_scores / math.sqrt(self.Wk.shape[1]), dim=-1)

        attn = einops.einsum(scores, v, "B N_q N_kv, B N_kv d_v -> B N_q d_v")
        return attn

def main():
    batch_size, n_tokens, d_in = 32, 64, 128
    x = torch.rand(batch_size, n_tokens, d_in)
    transformer = CausalTransformer(d_k=512, d_v=1028, d_in=d_in)
    return transformer(x)

if __name__ == "__main__":
    out = main()
    print(out.shape)
