import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..tl.validator import _validate_factors
from .utils import _set_up_cmap, _set_up_plot


def factor_embedding(
    adata: AnnData,
    obsm_key: str,
    annotation_key: str,
    factors: list[str] | str = "all",
    basis: str = "X_umap",
    cmap: str = "RdBu",
    colorbar_pos: str = "right",
    colorbar_width: str = "3%",
    orientation: str = "vertical",
    pad: float = 0.1,
    size: float = 1,
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    ax: Axes = None,
) -> Axes:
    """
    Plot factor weights on a given basis such as UMAP/TSNE.

    Parameters
    ----------
    adata
        AnnData object.
    model_key
        Key for the fitted model.
    factor
        Factor(s) to plot. If None, then all factors are plotted.
    basis
        Key for the basis (e.g. UMAP, T-SNE). If basis is None factor embedding
        tries to retrieve "X_{model_key}_umap".
    sign
        Sign of the factor. Should be either 1.0 or -1.0.
    cmap
        Colormap for the scatterplot.
    colorbar_pos
        Position of the colorbar.
    colorbar_width
        Width of the colorbar.
    orientation
        Orientation of the colorbar. Should be either "vertical" or "horizontal".
    pad
        Padding between the plot and colorbar
    size
        Marker/Dot size of the scatterplot.
    ncols
        Number of columns for the subplots.
    width
        Width of each subplot.
    height
        Height of each subplot.
    ax
        Axes object to plot on. If None, then a new figure is created. Works only
        if one factor is plotted.

    Returns
    -------
        Axes object.
    """
    num_factors = adata.obsm[obsm_key].shape[1]
    _validate_factors(factors, num_factors)

    ax = _set_up_plot(
        adata,
        obsm_key,
        factors,
        _factor_embedding,
        basis=basis,
        annotation_key=annotation_key,
        cmap=cmap,
        colorbar_pos=colorbar_pos,
        colorbar_width=colorbar_width,
        orientation=orientation,
        pad=pad,
        size=size,
        ncols=ncols,
        width=width,
        height=height,
        ax=ax,
    )
    return ax


def _factor_embedding(
    adata: AnnData,
    obsm_key: str,
    factor: int,
    basis: str,
    annotation_key: str,
    cmap: str,
    colorbar_pos: str,
    colorbar_width: str,
    orientation: str,
    pad: float,
    size: float,
    ax: Axes = None,
) -> Axes:
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = plt.gcf()

    weights = adata.obsm[obsm_key][..., int(factor)]
    cmap, norm = _set_up_cmap(weights, cmap)

    im = ax.scatter(
        adata.obsm[basis][:, 0],
        adata.obsm[basis][:, 1],
        s=size,
        c=weights,
        norm=norm,
        cmap=cmap,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(colorbar_pos, size=colorbar_width, pad=pad)
    fig.colorbar(im, cax=cax, orientation=orientation)
    if annotation_key in adata.uns:
        annot_dict = adata.uns[annotation_key]["mapping"]
        title = ""

        if f"{factor}-" in annot_dict:
            neg_axis = annot_dict[f"{factor}-"]
            title += f"{neg_axis} (-) vs "

        pos_axis = annot_dict[f"{factor}+"]
        title += f"{pos_axis} (+)"
        ax.set_title(title)

    else:
        ax.set_title(f"Factor {factor}")

    ax.set_xlabel(f"{basis}")
    ax.set_ylabel(f"{basis}")
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
