def count_model_params(seq_len, vocab_size, d_model, num_layers):
    """Given LM config calculate total number of parameters"""
    ffw_size = 4 * d_model  # in the number of intermediate features is always 4*d_model
    # token and position embeddings
    embeddings = d_model * vocab_size + d_model * seq_len
    # transformer blocks
    attention = 3 * d_model**2 + 3 * d_model  # weights and biases
    attproj = d_model**2 + d_model
    ffw = d_model * (ffw_size) + ffw_size
    ffwproj = ffw_size * d_model + d_model
    layernorms = 2 * 2 * d_model
    # dense
    ln_f = 2 * d_model
    dense = d_model * vocab_size  # note: no bias here
    # note: embeddings are not included in the param count!
    total_params = (
        num_layers * (attention + attproj + ffw + ffwproj + layernorms) + ln_f + dense
    )
    return total_params, embeddings


def calculate_training_steps(num_tokens, batch_size, block_size, epochs=1):
    """Given dataset size and training config, calculate number of training steps"""
    tokens_per_step = batch_size * block_size
    total_steps = (num_tokens * epochs) // tokens_per_step
    return total_steps
