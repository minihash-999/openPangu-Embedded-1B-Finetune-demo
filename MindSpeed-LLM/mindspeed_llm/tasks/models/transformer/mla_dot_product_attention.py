import math
from megatron.training import get_args
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from mindspeed_llm.core.models.common.embeddings.rotary_pos_embedding import yarn_get_mscale


class MlaDotProductAttention(DotProductAttention):
    """
    A special type of Dot Product Attention based on DotProductAttention.
    """

    def __init__(
            self,
            config: TransformerConfig,
            layer_number: int,
            attn_mask_type: AttnMaskType,
            attention_type: str,
            attention_dropout: float = None,
    ):
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout
        )
        args = get_args()

        self.scale_mask_softmax.scale = True
        self.hidden_size_per_partition = args.num_attention_heads * args.v_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)

        if args.rope_scaling_type is not None:
            mscale_all_dim = args.rope_scaling_mscale_all_dim if args.rope_scaling_mscale_all_dim else 0
            scaling_factor = args.rope_scaling_factor

            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.norm_factor = 1.0 / self.softmax_scale
