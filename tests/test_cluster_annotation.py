import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from langchain_community.llms.fake import FakeListLLM

from scllm.tl import annotate_cluster


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object with cluster labels."""
    # Create mock data with 3 clusters
    n_cells = 10
    n_genes = 5
    X = np.random.rand(n_cells, n_genes)
    obs = pd.DataFrame({"leiden": ["0", "0", "0", "1", "1", "1", "2", "2", "2", "2"]})
    var = pd.DataFrame(index=[f"gene{i+1}" for i in range(n_genes)])

    adata = sc.AnnData(X=X, obs=obs, var=var)

    # Add mock rank_genes_groups data
    adata.uns["leiden_rank_genes_groups"] = {
        "names": np.array(
            [
                ["gene1", "gene2", "gene3", "gene4", "gene5"],  # cluster 0
                ["gene2", "gene1", "gene4", "gene3", "gene5"],  # cluster 1
                ["gene3", "gene4", "gene1", "gene2", "gene5"],  # cluster 2
            ]
        ),
        "scores": np.array(
            [
                [1.0, 0.8, 0.6, 0.4, 0.2],  # cluster 0
                [0.9, 0.7, 0.5, 0.3, 0.1],  # cluster 1
                [0.8, 0.6, 0.4, 0.2, 0.0],  # cluster 2
            ]
        ),
        "pvals": np.array(
            [
                [0.01, 0.02, 0.03, 0.04, 0.05],  # cluster 0
                [0.02, 0.03, 0.04, 0.05, 0.06],  # cluster 1
                [0.03, 0.04, 0.05, 0.06, 0.07],  # cluster 2
            ]
        ),
        "pvals_adj": np.array(
            [
                [0.01, 0.02, 0.03, 0.04, 0.05],  # cluster 0
                [0.02, 0.03, 0.04, 0.05, 0.06],  # cluster 1
                [0.03, 0.04, 0.05, 0.06, 0.07],  # cluster 2
            ]
        ),
        "logfoldchanges": np.array(
            [
                [1.0, 0.8, 0.6, 0.4, 0.2],  # cluster 0
                [0.9, 0.7, 0.5, 0.3, 0.1],  # cluster 1
                [0.8, 0.6, 0.4, 0.2, 0.0],  # cluster 2
            ]
        ),
    }

    return adata


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    fake_responses = [
        '```json\n{"term": "T cell", "genes": ["CD4", "CD8A", "CD3E", "IL7R", "CD2"]}\n```',
        '```json\n{"term": "B cell", "genes": ["CD79A", "CD79B", "MS4A1", "CD19", "CD22"]}\n```',
        '```json\n{"term": "NK cell", "genes": ["NKG7", "GNLY", "PRF1", "GZMB", "KLRD1"]}\n```',
        '```json\n{"term": "Monocyte", "genes": ["CD14", "FCGR3A", "CD68", "CSF1R", "S100A8"]}\n```',
        '```json\n{"term": "Dendritic cell", "genes": ["CD1C", "CD1A", "CLEC9A", "CD207", "CD209"]}\n```',
    ]
    return FakeListLLM(responses=fake_responses)


def test_annotate_cluster_basic(mock_adata, mock_llm):
    """Test basic cluster annotation functionality."""
    annotate_cluster(mock_adata, "leiden", mock_llm, top_genes=3)
    print(mock_adata.obs)

    assert "cluster_annotation" in mock_adata.obs.columns
    assert "cluster_annotation" in mock_adata.uns
    assert isinstance(mock_adata.uns["cluster_annotation"], dict)


def test_annotate_cluster_custom_key(mock_adata, mock_llm):
    """Test cluster annotation with custom key."""
    custom_key = "custom_annotation"
    annotate_cluster(mock_adata, "leiden", mock_llm, key_added=custom_key, top_genes=3)

    assert custom_key in mock_adata.obs.columns
    assert f"custom_annotation" in mock_adata.uns


# def test_annotate_cluster_multiple_samples(mock_adata, mock_llm):
#     """Test cluster annotation with multiple samples."""
#     num_samples = 3
#     annotate_cluster(
#         mock_adata, "leiden", mock_llm, num_samples=num_samples, top_genes=3
#     )

#     # Check that we have multiple samples per cluster
#     samples_per_cluster = (
#         pd.DataFrame(mock_adata.uns["cluster_annotation"])
#         .groupby("cluster")["init"]
#         .nunique()
#     )
#     assert all(samples_per_cluster == num_samples)


def test_annotate_cluster_invalid_cluster_key(mock_adata, mock_llm):
    """Test cluster annotation with invalid cluster key."""
    with pytest.raises(KeyError):
        annotate_cluster(mock_adata, "invalid_key", mock_llm, top_genes=3)


# def test_annotate_cluster_empty_adata():
#     """Test cluster annotation with empty AnnData."""
#     empty_adata = sc.AnnData(
#         X=np.array([]),
#         obs=pd.DataFrame({"leiden": []}),
#         var=pd.DataFrame(index=[]),
#     )
#     with pytest.raises(ValueError):
#         annotate_cluster(empty_adata, "leiden", mock_llm)


def test_annotate_cluster_zero_top_genes(mock_adata, mock_llm):
    """Test cluster annotation with zero top genes."""
    with pytest.raises(ValueError):
        annotate_cluster(mock_adata, "leiden", mock_llm, top_genes=0)


def test_annotate_cluster_large_top_genes(mock_adata, mock_llm):
    """Test cluster annotation with top_genes larger than available genes."""
    with pytest.raises(ValueError):
        annotate_cluster(mock_adata, "leiden", mock_llm, top_genes=10)


def test_annotate_cluster_custom_term(mock_adata, mock_llm):
    """Test cluster annotation with custom term."""
    custom_term = "cell state"
    annotate_cluster(mock_adata, "leiden", mock_llm, term=custom_term, top_genes=3)

    assert "cluster_annotation" in mock_adata.obs.columns
    assert "cluster_annotation" in mock_adata.uns


def test_annotate_cluster_extra_prompt(mock_adata, mock_llm):
    """Test cluster annotation with extra prompt text."""
    extra = "Consider only immune cell types."
    annotate_cluster(mock_adata, "leiden", mock_llm, extra=extra, top_genes=3)

    assert "cluster_annotation" in mock_adata.obs.columns
    assert "cluster_annotation" in mock_adata.uns
