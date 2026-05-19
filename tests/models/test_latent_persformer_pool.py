"""Tests for LatentPersformer pooling refactor (M2).

Runs as a standalone script: ``uv run python tests/models/test_latent_persformer_pool.py``
or via pytest (no pytest fixtures needed; uses bare asserts).
"""

import torch

from src.models.latent_persformer import LatentPersformer


def _fixed_batch(seed: int = 0, batch: int = 3, seq: int = 16, d_in: int = 9):
    torch.manual_seed(seed)
    X = torch.randn(batch, seq, d_in)
    # Pad the last 4 positions of the last 2 samples to exercise the mask path.
    mask = torch.zeros(batch, seq, dtype=torch.bool)
    mask[1, -4:] = True
    mask[2, -2:] = True
    return X, mask


def _make_model(pooling: str, *, seed: int = 1234, **overrides):
    torch.manual_seed(seed)
    return LatentPersformer(
        d_in=9,
        d_out=2,
        d_model=32,
        d_latents=64,
        num_latents=16,
        num_blocks=1,
        num_self_attends_per_block=2,
        num_self_attention_heads=4,
        num_cross_attention_heads=4,
        dropout=0.0,
        decoder_hidden_dims=(32, 16),
        decoder_dropout=0.0,
        pooling=pooling,
        **overrides,
    )


def test_mean_pool_legacy_path_smoke():
    """``pooling='mean'`` should run cleanly and not allocate pool_attn parameters."""
    model = _make_model("mean")
    model.eval()
    X, mask = _fixed_batch()
    with torch.no_grad():
        y = model(X, mask)
    assert y.shape == (3, 2), f"unexpected logits shape: {y.shape}"
    assert model.pool_query is None
    assert model.pool_attn is None
    # No mean-pool-mode parameters named pool_*.
    pool_params = [n for n, _ in model.named_parameters() if n.startswith("pool_")]
    assert pool_params == [], f"mean mode leaked pool params: {pool_params}"


def test_attn_pool_smoke():
    """``pooling='attn'`` builds pool_query + pool_attn and runs end-to-end."""
    model = _make_model("attn")
    model.eval()
    X, mask = _fixed_batch()
    with torch.no_grad():
        y = model(X, mask)
    assert y.shape == (3, 2)
    assert model.pool_query is not None
    assert model.pool_query.shape == (1, 1, 64)
    assert model.pool_attn is not None


def test_aux_off_by_default():
    """No aux collection unless explicitly enabled — keeps training overhead at zero."""
    model = _make_model("attn")
    model.eval()
    X, mask = _fixed_batch()
    with torch.no_grad():
        _ = model(X, mask)
    assert model.aux == {}


def test_aux_collection_shapes():
    """When aux is enabled, cross_attn has shape (B, cross_heads, num_latents, seq) and rows sum to 1."""
    model = _make_model("attn")
    model.eval()
    model.enable_aux(True)
    X, mask = _fixed_batch()
    with torch.no_grad():
        _ = model(X, mask)

    aux = model.aux
    assert set(aux.keys()) == {"latents", "cross_attn", "self_attn"}

    latents = aux["latents"]
    assert latents.shape == (3, 16, 64), f"latents shape: {latents.shape}"

    cross = aux["cross_attn"]
    assert cross is not None and len(cross) == 1, "expected one cross-attention block"
    ca = cross[-1]
    assert ca.shape == (3, 4, 16, 16), f"cross_attn shape: {ca.shape}"
    row_sums = ca.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        f"cross_attn rows should sum to ~1, got min={row_sums.min()}, max={row_sums.max()}"
    )

    self_attn = aux["self_attn"]
    assert self_attn is not None and len(self_attn) == 2, (
        f"expected 2 self-attention steps (1 block * 2 self_attends), got {len(self_attn)}"
    )
    sa = self_attn[-1]
    assert sa.shape == (3, 4, 16, 16), f"self_attn shape: {sa.shape}"


def test_aux_disable_clears_state():
    model = _make_model("attn")
    model.enable_aux(True)
    X, mask = _fixed_batch()
    with torch.no_grad():
        _ = model(X, mask)
    assert model.aux != {}
    model.enable_aux(False)
    assert model.aux == {}


def test_mean_path_unchanged_under_pool_query_absence():
    """Regression: mean-pool forward pass must be reproducible from same seed.

    Pre-refactor code path: linear_in -> Perceiver -> latents.mean(dim=1) -> decoder.
    Post-refactor with pooling='mean': identical control flow, identical params
    (no pool_query / pool_attn allocated when pooling='mean').
    """
    X, mask = _fixed_batch()

    m1 = _make_model("mean", seed=99)
    m1.eval()
    with torch.no_grad():
        y1 = m1(X, mask)

    m2 = _make_model("mean", seed=99)
    m2.eval()
    with torch.no_grad():
        y2 = m2(X, mask)

    assert torch.allclose(y1, y2, atol=0, rtol=0), "mean-pool path not deterministic under same seed"


def test_pooling_invalid_raises():
    try:
        _make_model("max")
    except ValueError as e:
        assert "pooling" in str(e)
        return
    raise AssertionError("expected ValueError for invalid pooling mode")


def test_d_latents_not_divisible_by_pool_heads():
    # d_latents=64 keeps PerceiverConfig happy (64 % 8 == 0); pooling_heads=7 must fail in our check.
    try:
        LatentPersformer(d_latents=64, num_self_attention_heads=8, num_cross_attention_heads=8, pooling="attn", pooling_heads=7)
    except ValueError as e:
        assert "pooling_heads" in str(e)
        return
    raise AssertionError("expected ValueError for pooling_heads not dividing d_latents")


if __name__ == "__main__":
    tests = [
        ("mean_pool_legacy_path_smoke", test_mean_pool_legacy_path_smoke),
        ("attn_pool_smoke", test_attn_pool_smoke),
        ("aux_off_by_default", test_aux_off_by_default),
        ("aux_collection_shapes", test_aux_collection_shapes),
        ("aux_disable_clears_state", test_aux_disable_clears_state),
        ("mean_path_unchanged_under_pool_query_absence", test_mean_path_unchanged_under_pool_query_absence),
        ("pooling_invalid_raises", test_pooling_invalid_raises),
        ("d_latents_not_divisible_by_pool_heads", test_d_latents_not_divisible_by_pool_heads),
    ]
    fails = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e!r}")
            fails.append(name)
    if fails:
        raise SystemExit(f"{len(fails)} test(s) failed: {fails}")
    print(f"\nAll {len(tests)} tests passed.")
