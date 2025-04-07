import torch
import torch.nn as nn

from llm_from_scratch.GPT_to_Llama32.llama_transformer_block import RMSNorm
from llm_from_scratch.rope import RoPE


class MultiLatentAttention(nn.Module):
    """
    This is the MLA training implementation (not inference) purely based on DeepSeekV2 and DeepSeekV3 research papers.

    This class performs multi-head attention with a focus on latent spaces for queries and keys/values.
    It incorporates decoupling for RoPE and additional RMS normalization for stability.

    Args:
        d_in (int): Input embedding dimension.
        d_out (int): Output embedding dimension (must be divisible by num_heads).
        ctx_len (int): Maximum context/sequence length.
        num_heads (int): Number of attention heads.
        rope_base (int, optional): Base for RoPE. Defaults to 500_000.
        rope_cfg (dict, optional): Configuration for RoPE. Defaults to None.

    note: d_out must be divisible by num_heads, decoup_head_dim must be be divisible by 2 for RoPE.
            A good starting point is choosing d_out / num_heads = being a power of 2
    """

    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
    ):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // self.num_heads
        self.q_rank = 1536  # TODO they hardcoded, whats the rationale? adapt for small models?
        self.kv_rank = 4 * self.head_dim
        self.decoup_head_dim = self.head_dim // 2
        self.att_scaling = (self.head_dim + self.decoup_head_dim) ** -0.5
        self.out_proj = nn.Linear(d_out, d_out)

        # up and down projection layers
        self.wq_down_proj = nn.Linear(d_in, self.q_rank)
        self.wq_up_proj = nn.Linear(self.q_rank, d_out)
        self.wq_decoup = nn.Linear(self.q_rank, num_heads * self.decoup_head_dim)
        self.wkv_down_proj = nn.Linear(d_in, self.kv_rank)
        self.wk_up_proj = nn.Linear(self.kv_rank, d_out)
        self.wv_up_proj = nn.Linear(self.kv_rank, d_out)
        self.wk_decoup = nn.Linear(d_in, self.decoup_head_dim)

        # additional norm for stability, per DeepSeekV3 4.2 Hparams
        self.q_rms_norm = RMSNorm(self.q_rank)
        self.kv_rms_norm = RMSNorm(self.kv_rank)

    def forward(self, x, mask, cos, sin):
        b, seq_len, d_in = x.shape

        # down projection
        q_latent = self.wq_down_proj(x)
        q_latent = self.q_rms_norm(q_latent)
        kv_latent = self.wkv_down_proj(x)
        kv_latent = self.kv_rms_norm(kv_latent)

        # up projection
        queries = self.wq_up_proj(q_latent)
        keys = self.wk_up_proj(kv_latent)
        values = self.wv_up_proj(kv_latent)

        # reshaping
        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(b, seq_len, self.num_heads, -1)
        values = values.view(b, seq_len, self.num_heads, -1)
        queries = torch.transpose(queries, 1, 2)
        keys = keys.transpose(1, 2)

        # decoupling and reshaping projections for RoPE
        decoup_queries = self.wq_decoup(q_latent)
        decoup_queries = decoup_queries.view(b, seq_len, -1, self.decoup_head_dim)
        decoup_queries = torch.transpose(decoup_queries, 1, 2)
        decoup_keys = self.wk_decoup(x)
        decoup_keys = decoup_keys.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        decoup_keys = decoup_keys.transpose(1, 2)
        # RoPE
        decoup_queries = RoPE.apply(decoup_queries, cos, sin)
        decoup_keys = RoPE.apply(decoup_keys, cos, sin)

        # concatenating back decoupled queries and keys
        queries = torch.cat([queries, decoup_queries], dim=-1)
        keys = torch.cat([keys, decoup_keys], dim=-1)

        # attention
        att_scores = queries @ keys.mT
        # mask up to seq length/num of tokens
        current_mask = mask.bool()[:seq_len, :seq_len]
        # scaling by √(head_dim + decoup_head_dim)
        scaled_att_scores = att_scores * self.att_scaling
        # masking in place and normalizing with softmax
        scaled_att_scores.masked_fill_(current_mask, -torch.inf)
        att_weights = torch.softmax(scaled_att_scores, dim=-1)

        values = values.transpose(1, 2)
        ctx_tensor = att_weights @ values
        ctx_tensor = ctx_tensor.transpose(1, 2)
        ctx_tensor = ctx_tensor.contiguous().view(b, seq_len, self.d_out)

        ctx_tensor = self.out_proj(ctx_tensor)

        return ctx_tensor


# testing
if __name__ == "__main__":
    torch.manual_seed(123)

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your (x^1)
            [0.55, 0.87, 0.66],  # journey (x^2)
            [0.57, 0.85, 0.64],  # starts (x^3)
            [0.22, 0.58, 0.33],  # with (x^4)
            [0.77, 0.25, 0.10],  # one (x^5)
            [0.05, 0.80, 0.55],  # step (x^6)
        ]
    )

    d_in = inputs.shape[-1]
    d_out = 32

    input_batch = torch.stack((inputs, inputs), dim=0)

    mla = MultiLatentAttention(d_in, d_out, 6, 4)
    print(mla(input_batch))
