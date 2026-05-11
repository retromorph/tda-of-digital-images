from exp.runners._shim import run_legacy_shim


if __name__ == "__main__":
    run_legacy_shim(
        model_name="MLP",
        input_kind="flat",
        model_arg_keys={"d_hidden", "num_layers", "dropout", "activation", "alpha"},
    )
