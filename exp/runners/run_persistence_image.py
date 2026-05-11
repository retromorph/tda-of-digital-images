from exp.runners._shim import run_legacy_shim


if __name__ == "__main__":
    run_legacy_shim(
        model_name="PERSISTENCE_CNN2D",
        input_kind="encoded",
        encoder_name="persistence_image",
        model_arg_keys={"base_channels", "dropout", "in_channels"},
        encoder_arg_keys={"resolution", "sigma2", "weighting", "weight_power"},
    )
