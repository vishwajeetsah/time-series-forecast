"""Optional matplotlib helpers. Importing this module requires matplotlib."""

import numpy as np


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib: pip install matplotlib") from exc


def plot_forecast(y, forecast, train_end=None, path="forecast.png", title="Forecast"):
    """Plot history + h-step forecast, saving to `path`."""
    plt = _plt()
    y = np.asarray(y, dtype=float)
    n = len(y)
    h = len(forecast)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(n), y, label="history")
    ax.plot(np.arange(n, n + h), forecast, label="forecast", linestyle="--")
    if train_end is not None:
        ax.axvline(train_end, color="grey", linestyle=":", label="train/test split")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_decomposition(y, components, path="decomposition.png"):
    """Plot observed series plus trend/seasonal/residual panels."""
    plt = _plt()
    y = np.asarray(y, dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, (label, vals) in zip(
        axes,
        [
            ("observed", y),
            ("trend", components["trend"]),
            ("seasonal", components["seasonal"]),
            ("residual", components["resid"]),
        ],
    ):
        ax.plot(vals)
        ax.set_ylabel(label)
    fig.suptitle(f"Additive decomposition (m={components['m']})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
