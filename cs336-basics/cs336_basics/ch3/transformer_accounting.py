
def calculate_parameters(
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int
) -> int:
    embedding_block_params = vocab_size * d_model
    norm_params = d_model
    multihead_attention_params = 4 * d_model * d_model
    ff_params = 3 * d_ff * d_model
    transformer_block_params = norm_params + multihead_attention_params + norm_params + ff_params
    output_linear_params = d_model * vocab_size
    transformer_lm_params = embedding_block_params + transformer_block_params * num_layers + norm_params + output_linear_params
    byte = transformer_lm_params * 4
    GB = byte / 1024 / 1024 / 1024

    print(f"params:{transformer_lm_params}\nmemory: {GB} GB")

    return transformer_lm_params


def mat_mul(s1:int, s2:int, s3:int):
    return 2 * s1 * s2 * s3


def calculate_flops(
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int
):
    embedding_flops = 0
    norm_flops = 0
    position_wise_ff_flops = mat_mul(d_ff, d_model, context_length) * 3
    attention_all_flops, qkv_flops, attention_flops = calculate_multihead_attention_flops(context_length, d_model, num_heads)
    transformer_block_flops = attention_all_flops + position_wise_ff_flops
    output_linear_flops = mat_mul(context_length, d_model, vocab_size)
    transformer_lm_flops = num_layers * transformer_block_flops + output_linear_flops

    print(f"all flops: {transformer_lm_flops / 1024/1024/1024/1024} TFlops")
    print(f"single qkv_flops: {qkv_flops / 1024/1024/1024} GFlops\n"
          f"single attention_flops: {attention_flops/1024/1024/1024} GFlops\n"
          f"single ff_flops: {position_wise_ff_flops/1024/1024/1024} GFlops")

def calculate_multihead_attention_flops(
        context_length: int,
        d_model: int,
        num_heads: int,
) -> tuple:
    # Q K V
    qkv = 3 * mat_mul(context_length, d_model, d_model)
    d_k = d_model // num_heads
    attention = num_heads * (mat_mul(context_length, d_k, context_length) + mat_mul(context_length, context_length, d_k))
    output = mat_mul(context_length, d_model, d_model)
    return qkv + attention + output, qkv, attention


def calculate_parameters_and_flops(
        name: str,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int
):
    print(f"===== {name} ====")
    calculate_parameters(vocab_size, context_length, num_layers, d_model, num_heads, d_ff)
    calculate_flops(vocab_size, context_length, num_layers, d_model, num_heads, d_ff)
    print("\n")


def calculate_dff(d_model: int) -> int:
    return int(d_model * 8 / 3) // 64 * 64

# calculate_parameters_and_flops("XL", 50257, 1024, 48, 1600, 25, 4288)
# calculate_parameters_and_flops("small", 50257, 1024, 12, 768, 12, calculate_dff(768))
# calculate_parameters_and_flops("medium", 50257, 1024, 24, 1024, 16, calculate_dff(1024))
# calculate_parameters_and_flops("large", 50257, 1024, 36, 1280, 20, calculate_dff(1280))
#
# calculate_parameters_and_flops("XL long context", 50257, 16384, 48, 1600, 25, 4288)