from textwrap import wrap

import pandas as pd
import scanpy as sc
from joypy import joyplot

from ..tl.validator import _validate_factors


def factor_ridge(
    adata: sc.AnnData,
    obsm_key: str,
    annotation_key: str,
    factors: list[str] | str = "all",
    **kwargs,
):

    num_factors = adata.obsm[obsm_key].shape[1]
    factors = _validate_factors(factors, num_factors)

    data = pd.DataFrame(
        adata.obsm[obsm_key],
        columns=[str(i) for i in range(num_factors)],
    ).loc[:, factors]

    fig, axes = joyplot(data, **kwargs)
    factor_axes = {int(f): i for i, f in enumerate(factors)}

    for k, v in adata.uns[annotation_key]["mapping"].items():
        factor, sign = k[:-1], k[-1]
        if factor not in factors:
            continue

        factor = int(factor)
        ax = axes[factor_axes[factor]]
        if sign == "+":
            ax.text(
                0.99,
                0.2,
                "\n".join(wrap(v, 20)),
                ha="right",
                fontsize="small",
                color="k",
                va="center",
                transform=ax.transAxes,
            )
        if sign == "-":
            ax.text(
                0.01,
                0.2,
                "\n".join(wrap(v, 20)),
                ha="left",
                fontsize="small",
                color="k",
                va="center",
                transform=ax.transAxes,
            )

    return axes
