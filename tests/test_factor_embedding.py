import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scllm.pl import factor_embedding


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object with factor embeddings and UMAP coordinates."""
    # Create mock data with 3 factors
    n_cells = 10
    n_genes = 5
    X = np.random.rand(n_cells, n_genes)
    obs = pd.DataFrame({"leiden": ["0", "0", "0", "1", "1", "1", "2", "2", "2", "2"]})
    var = pd.DataFrame(index=[f"gene{i+1}" for i in range(n_genes)])

    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.varm["pca"] = np.random.rand(n_genes, 3)
    # Add mock factor embeddings
    adata.obsm["X_pca"] = np.random.rand(n_cells, 3)

    # Add mock UMAP coordinates
    adata.obsm["X_umap"] = np.random.rand(n_cells, 2)

    # Add mock factor annotations
    adata.uns["factor_annotation"] = {
        "mapping": {
            "0+": "T cell",
            "0-": "B cell",
            "1+": "NK cell",
            "1-": "Monocyte",
            "2+": "Dendritic cell",
            "2-": "Macrophage",
        }
    }

    return adata


def test_factor_embedding_basic(mock_adata):
    """Test basic factor embedding functionality."""
    ax = factor_embedding(
        mock_adata,
        "X_pca",
        annotation_key="factor_annotation",
        factors=["0", 1],
        basis="X_umap",
    )

    assert isinstance(ax[0], plt.Axes)
    assert len(ax.flatten()) == 2  # Two subplots for two factors


def test_factor_embedding_single_factor(mock_adata):
    """Test factor embedding with single factor."""
    ax = factor_embedding(
        mock_adata,
        "X_pca",
        annotation_key="factor_annotation",
        factors=["0"],
        basis="X_umap",
    )

    assert isinstance(ax, plt.Axes)
    # assert len(ax.figure.axes) == 1  # Single subplot
    # assert ax.get_xlabel() == "X_umap"
    # assert ax.get_ylabel() == "X_umap"


def test_factor_embedding_custom_basis(mock_adata):
    """Test factor embedding with custom basis."""
    # Add custom basis
    mock_adata.obsm["X_custom"] = np.random.rand(mock_adata.n_obs, 2)

    ax = factor_embedding(
        mock_adata,
        "X_pca",
        annotation_key="factor_annotation",
        factors=0,
        basis="X_custom",
    )

    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "X_custom"
    assert ax.get_ylabel() == "X_custom"


def test_factor_embedding_custom_cmap(mock_adata):
    """Test factor embedding with custom colormap."""
    ax = factor_embedding(
        mock_adata,
        "X_pca",
        annotation_key="factor_annotation",
        factors=0,
        basis="X_umap",
        cmap="viridis",
    )

    assert isinstance(ax, plt.Axes)
    scatter = ax.collections[0]
    # print(scatter)
    # assert scatter.get_cmap().name == "viridis"


def test_factor_embedding_custom_size(mock_adata):
    """Test factor embedding with custom point size."""
    size = 5
    ax = factor_embedding(
        mock_adata,
        "X_pca",
        annotation_key="factor_annotation",
        factors=["0"],
        basis="X_umap",
        size=size,
    )

    assert isinstance(ax, plt.Axes)
    scatter = ax.collections[0]
    assert scatter.get_sizes()[0] == size


def test_factor_embedding_custom_layout(mock_adata):
    """Test factor embedding with custom layout parameters."""
    ncols = 2
    width = 6
    height = 4

    ax = factor_embedding(
        mock_adata,
        "X_pca",
        annotation_key="factor_annotation",
        factors=["0", "1", "2"],
        basis="X_umap",
        ncols=ncols,
        width=width,
        height=height,
    )

    assert isinstance(ax[0, 0], plt.Axes)
    assert ax[0, 0].figure.get_figwidth() == width * 2  # 2 columns
    assert ax[0, 0].figure.get_figheight() == height * 2  # 2 rows


def test_factor_embedding_custom_annotation(mock_adata):
    """Test factor embedding with custom annotation key."""
    custom_key = "custom_annotation"
    mock_adata.uns[custom_key] = mock_adata.uns["factor_annotation"]

    ax = factor_embedding(
        mock_adata, "X_pca", factors=["0"], basis="X_umap", annotation_key=custom_key
    )

    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "B cell (-) vs T cell (+)"  # Check title from annotation


def test_factor_embedding_no_annotation(mock_adata):
    """Test factor embedding without annotations."""
    # Remove annotations
    del mock_adata.uns["factor_annotation"]

    ax = factor_embedding(
        mock_adata,
        "X_pca",
        annotation_key="factor_annotation",
        factors=0,
        basis="X_umap",
    )

    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Factor 0"  # Default title without annotation


def test_factor_embedding_invalid_basis(mock_adata):
    """Test factor embedding with invalid basis."""
    with pytest.raises(KeyError):
        factor_embedding(
            mock_adata,
            "X_pca",
            annotation_key="factor_annotation",
            factors=0,
            basis="invalid_basis",
        )


def test_factor_embedding_invalid_factor(mock_adata):
    """Test factor embedding with invalid factor."""
    with pytest.raises(IndexError):
        factor_embedding(
            mock_adata,
            "X_pca",
            annotation_key="factor_annotation",
            factors=10,
            basis="X_umap",
        )


def test_factor_embedding_custom_ax(mock_adata):
    """Test factor embedding with custom axes."""
    fig, ax = plt.subplots()
    result_ax = factor_embedding(
        mock_adata,
        "X_pca",
        annotation_key="factor_annotation",
        factors=["0"],
        basis="X_umap",
        ax=ax,
    )

    # assert result_ax == ax
    assert len(ax.figure.axes) == 1  # Single axes
    # assert ax.get_xlabel() == "X_umap"
    # assert ax.get_ylabel() == "X_umap"
