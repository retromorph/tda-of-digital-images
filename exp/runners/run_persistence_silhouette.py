from exp.runners._shim import run_legacy_shim


if __name__ == "__main__":
    run_legacy_shim(
        model_name="PERSISTENCE_CNN1D",
        input_kind="encoded",
        encoder_name="persistence_silhouette",
        model_arg_keys={"base_channels", "dropout", "in_channels"},
        encoder_arg_keys={"resolution", "weighting", "weight_power"},
    )
