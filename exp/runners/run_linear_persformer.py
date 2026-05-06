from exp.runners._shim import run_legacy_shim


if __name__ == "__main__":
    run_legacy_shim(
        model_name="LINEAR_PERSFORMER",
        input_kind="diagram",
        model_arg_keys={
            "d_model",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_landmarks",
            "encoder_dropout",
            "decoder_dropout",
            "decoder_hidden_dims",
            "pooling_heads",
            "activation",
        },
    )
