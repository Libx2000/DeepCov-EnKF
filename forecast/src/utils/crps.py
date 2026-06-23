import numpy as np

def _get_n_ensemble(
    ds: np.ndarray,
    expect_n_ensemble_at_least: int = 1,
) -> int:
    """Returns the size of `ensemble_dim`, optionally asserting size at least."""
    n_ensemble = ds.shape[0]
    if n_ensemble < expect_n_ensemble_at_least:
        raise ValueError(
            f"{n_ensemble=} is less than expected size of "
            f"{expect_n_ensemble_at_least}"
        )
    return n_ensemble

def _pointwise_crps_spread(
    forecast: np.ndarray
) -> np.ndarray:
    """CRPS spread at each point in truth, averaged over ensemble only."""
    n_ensemble = _get_n_ensemble(forecast)

    # one_half_spread is ̂̂λ₂ from Zamo. That is, with n_ensemble = M,
    #   λ₂ = 1 / (2 M (M - 1)) Σ_{i,j=1}^M |Xi - Xj|
    # See the definition of eFAIR and then
    # eqn 3 (appendix B), which shows that this double summation of absolute
    # differences can be written as a sum involving sorted elements multiplied
    # by their index. That is, if X1 < X2 < ... < XM,
    #   λ₂ = 1 / (M(M-1)) Σ_{i,j=1}^M (2*i - M - 1) Xi.
    # The term (2*i - M - 1) is +1 times the number of terms Xi is greater than,
    # and -1 times the number of terms Xi is less than.
    # Here we do not sort, but instead compute the rank of each element, multiply
    # appropriately, then sum. We prefer this second form, since it involves an
    # O(M Log[M]) compute and O(M) memory usage, whereas the first is O(M²) in
    # compute and memory.
    rank = _rank_ds(forecast)
    return (
        2
        * (
            np.nanmean((2 * rank - n_ensemble - 1) * forecast)
        )
        / (n_ensemble - 1)
  )
  
def _pointwise_crps_skill(
    forecast: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    """CRPS skill at each point in truth, averaged over ensemble only."""
    _get_n_ensemble(forecast)  # Will raise if no ensembles.
    return np.nanmean(np.abs(truth - forecast), axis=0)

def _rank_ds(ds: np.ndarray) -> np.ndarray:
    """The ranking of `ds` along `dim`, with 1 being the smallest entry."""

    def _rank_da(da: np.ndarray) -> np.ndarray:
        return _rankdata(da.values, axis=0)

    return ds.copy(data={k: _rank_da(v) for k, v in ds.items()})

def _rankdata(x: np.ndarray, axis: int) -> np.ndarray:
    """Version of (ordinal) scipy.rankdata from V13."""
    x = np.asarray(x)
    x = np.swapaxes(x, axis, -1)
    j = np.argsort(x, axis=-1)
    ordinal_ranks = np.broadcast_to(
        np.arange(1, x.shape[-1] + 1, dtype=int), x.shape
    )
    ordered_ranks = np.empty(j.shape, dtype=ordinal_ranks.dtype)
    np.put_along_axis(ordered_ranks, j, ordinal_ranks, axis=-1)
    return np.swapaxes(ordered_ranks, axis, -1)

def _spatial_average(
    dataset: np.ndarray, region: type.Optional[Region] = None, skipna: bool = False
) -> np.ndarray:
    """Compute spatial average after applying region mask.

    Args:
        dataset: Metric dataset as a function of latitude/longitude.
        region: Region object (optional).
        skipna: Skip NaNs in spatial mean.

    Returns:
        dataset: Spatially averaged metric.
    """
    weights = get_lat_weights(dataset)
    if region is not None:
        dataset, weights = region.apply(dataset, weights)
        # ignore NaN/Inf values in regions with zero weight
        dataset = dataset.where(weights > 0, 0)
    return dataset.weighted(weights).mean(
        ["latitude", "longitude"], skipna=skipna
    )

def crps_skill(
    forecast: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    """CRPS skill at each point in truth, averaged over ensemble only."""
    return _pointwise_crps_skill(forecast, truth)