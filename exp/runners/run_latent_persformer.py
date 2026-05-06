from exp.runners._shim import run_legacy_shim


if __name__ == "__main__":
    run_legacy_shim(
        model_name="LATENT_PERSFORMER",
        input_kind="diagram",
        model_arg_keys={
            "d_model",
            "d_latents",
            "num_latents",
            "num_blocks",
            "num_self_attends_per_block",
            "num_self_attention_heads",
            "num_cross_attention_heads",
            "cross_attention_widening_factor",
            "self_attention_widening_factor",
            "dropout",
            "decoder_dropout",
            "decoder_hidden_dims",
            "activation",
        },
    )
