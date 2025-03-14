from textwrap import wrap
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

from ..tl.validator import _validate_factors


def factor_stripplot(
    adata: sc.AnnData,
    obsm_key: str,
    annotation_key: str,
    factors: List[str] | str = "all",
    sign: str = "+",
    ax: plt.Axes = None,
    **kwargs,
):

    num_factors = adata.obsm[obsm_key].shape[1]
    factors = _validate_factors(factors, num_factors)

    if ax is None:
        ax = plt.gca()

    data = (
        pd.DataFrame(
            adata.obsm[obsm_key],
            columns=[str(i) for i in range(num_factors)],
        )
        .loc[:, factors]
        .melt(var_name="Factor", value_name="Weight")
    )

    sns.stripplot(data=data, y="Factor", x="Weight", hue="Weight", ax=ax, **kwargs)

    for k, v in adata.uns[annotation_key]["mapping"].items():
        factor, sign = k[:-1], k[-1]
        if factor not in factors:
            continue
        factor = int(factor)
        if sign == "+":
            ax.text(
                0.99,
                factor,
                "\n".join(wrap(v, 20)),
                ha="right",
                fontsize="small",
                color="k",
                va="center",
                transform=ax.get_yaxis_transform(),
            )
        if sign == "-":
            ax.text(
                0.01,
                factor,
                "\n".join(wrap(v, 20)),
                ha="left",
                fontsize="small",
                color="k",
                va="center",
                transform=ax.get_yaxis_transform(),
            )
