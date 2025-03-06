from textwrap import wrap
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns


def factor_stripplot(
    adata: sc.AnnData,
    obsm_key: str,
    annotation_key: str,
    factors: List[str] | str = "all",
    sign: str = "+",
    ax: plt.Axes = None,
    **kwargs,
):
    # TODO refactor (also used by annotate factor)
    if isinstance(factors, str):
        if factors == "all":
            factors = [str(i) for i in range(adata.obsm[obsm_key].shape[1])]
        else:
            factors = [factors]

    if ax is None:
        ax = plt.gca()

    data = (
        pd.DataFrame(
            adata.obsm[obsm_key],
            columns=[str(i) for i in range(adata.obsm[obsm_key].shape[1])],
        )
        .loc[:, factors]
        .melt(var_name="Factor", value_name="Weight")
    )

    sns.stripplot(data=data, y="Factor", x="Weight", ax=ax, **kwargs)

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
