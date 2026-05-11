from exp.runners._shim import run_legacy_shim


if __name__ == "__main__":
    run_legacy_shim(
        model_name="ViT",
        input_kind="image",
        model_arg_keys={"d_model", "d_hidden", "n_heads", "n_blocks", "patch_size", "dropout"},
    )
