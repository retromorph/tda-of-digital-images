from exp.runners._shim import run_legacy_shim


if __name__ == "__main__":
    run_legacy_shim(
        model_name="PERSFORMER",
        input_kind="diagram",
        model_arg_keys={
            "d_model",
            "d_hidden",
            "num_heads",
            "num_layers",
            "encoder_dropout",
            "decoder_dropout",
            "decoder_hidden_dims",
            "pooling_heads",
            "norm",
            "activation",
            "alpha",
            "norm_first",
        },
    )
