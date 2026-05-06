from exp.runners._shim import run_legacy_shim


if __name__ == "__main__":
    run_legacy_shim(
        model_name="PHTS",
        input_kind="diagram",
        model_arg_keys={"d_model", "d_hidden", "dropout", "activation", "alpha"},
    )
