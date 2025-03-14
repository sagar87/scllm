import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from langchain_community.llms.fake import FakeListLLM

from scllm.tl import annotate_factor


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object with factor loadings."""
    # Create mock factor loadings with known values
    n_genes = 5
    n_factors = 3
    loadings = np.array(
        [
            [1.0, 0.5, -0.3],  # gene1
            [0.8, -0.7, 0.2],  # gene2
            [0.6, 0.9, -0.5],  # gene3
            [0.4, -0.4, 0.8],  # gene4
            [0.2, 0.3, -0.9],  # gene5
        ]
    )

    adata = sc.AnnData(
        X=np.random.rand(10, n_genes),  # Random expression data
        var=pd.DataFrame(index=[f"gene{i+1}" for i in range(n_genes)]),
    )

    adata.obsm["X_pca"] = np.random.rand(10, n_factors)
    adata.varm["pca"] = loadings
    return adata


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    fake_responses = [
        '```json\n{"term": "Monocyte/Macrophage", "genes": ["CST3", "TYROBP", "FCN1", "LST1", "AIF1", "S100A8", "TYMP", "LGALS2", "FCER1G", "LGALS1"]}\n```',
        '```json\n{"term": "Cytotoxic T cell", "genes": ["NKG7", "GZMB", "PRF1", "CST7", "GZMA", "FGFBP2", "GNLY", "CTSW", "SPON2", "CCL4"]}\n```',
        '```json\n{"term": "Platelet", "genes": ["PF4", "PPBP", "SDPR", "SPARC", "GNG11", "HIST1H2AC", "GP9", "NRGN", "TUBB1", "RGS18"]}\n```',
        '```json\n{"term": "B cell", "genes": ["CD79A", "HLA-DQA1", "CD79B", "MS4A1", "HLA-DQB1", "HLA-DPB1", "HLA-DPA1", "HLA-DRB1", "TCL1A", "LINC00926"]}\n```',
        '```json\n{\n  "term": "Monocyte",\n  "genes": [\n    "FCGR3A",\n    "CTD-2006K23.1",\n    "IFITM3",\n    "ABI3",\n    "CEBPB",\n    "C1QA",\n    "TNFSF10",\n    "TNFRSF1B",\n    "SMPDL3A",\n    "LYPD2"\n  ]\n}\n```',
        '```json\n{"term": "Cytotoxic T cell", "genes": ["GZMK", "CCL5", "LYAR", "IL32", "KLRG1", "TIGIT", "JAKMIP1", "GZMA", "CD2", "GPR171"]}\n```',
        '```json\n{"term": "T helper 1 (Th1) cell", "genes": ["SELL", "LTB", "TNFRSF25", "TNFRSF4", "ICOS", "MAL", "IGFBP7", "KIF5B", "PLBD1", "AKR1C3"]}\n```',
        '```json\n{\n  "term": "Natural Killer T (NKT) cells",\n  "genes": ["KLRC1", "XCL1", "HOPX", "XCL2", "NCR3", "GZMK", "KLRB1", "PRR5", "SPTSSB", "S100A12"]\n}\n```',
        '```json\n{"term": "Activated T cell", "genes": ["IFIT1", "TREX1", "GZMH", "IFI35", "HEXIM2", "GBP1", "SLC40A1", "TBPL1", "TIGIT", "FCRL6"]}\n```',
        '```json\n{"term": "Natural Killer (NK) Cell", "genes": ["XCL1", "XCL2", "IFIT1", "APOBEC3B", "TNFSF10", "KLRC1", "SELL", "OASL", "GADD45B", "SEPHS2"]}\n```',
    ]

    # Create the faker
    return FakeListLLM(responses=fake_responses)


def test_annotate_factor_basic(mock_adata, mock_llm):
    """Test basic factor annotation functionality."""
    annotate_factor(mock_adata, "pca", mock_llm, factors=["0", "1"], top_genes=3)

    assert "factor_annotation" in mock_adata.uns
    assert isinstance(mock_adata.uns["factor_annotation"], dict)
    assert "raw" in mock_adata.uns["factor_annotation"]
    assert "mapping" in mock_adata.uns["factor_annotation"]


def test_annotate_factor_custom_key(mock_adata, mock_llm):
    """Test factor annotation with custom key."""
    custom_key = "custom_annotation"
    annotate_factor(mock_adata, "pca", mock_llm, key_added=custom_key, top_genes=3)

    assert custom_key in mock_adata.uns
    assert isinstance(mock_adata.uns[custom_key], dict)


def test_annotate_factor_single_factor(mock_adata, mock_llm):
    """Test factor annotation with single factor."""
    annotate_factor(mock_adata, "pca", mock_llm, factors="0", top_genes=3)

    assert "factor_annotation" in mock_adata.uns
    assert len(mock_adata.uns["factor_annotation"]["mapping"]) == 2  # One factor


def test_annotate_factor_all_factors(mock_adata, mock_llm):
    """Test factor annotation with all factors."""
    annotate_factor(mock_adata, "pca", mock_llm, factors="all", top_genes=3)

    assert "factor_annotation" in mock_adata.uns
    assert len(mock_adata.uns["factor_annotation"]["mapping"]) == 6  # All factors


def test_annotate_factor_multiple_samples(mock_adata, mock_llm):
    """Test factor annotation with multiple samples."""
    num_samples = 3
    annotate_factor(mock_adata, "pca", mock_llm, num_samples=num_samples, top_genes=3)

    # Check that we have multiple samples per factor
    samples_per_factor = (
        pd.DataFrame(mock_adata.uns["factor_annotation"]["raw"])
        .groupby("factor")["init"]
        .nunique()
    )
    assert all(samples_per_factor == num_samples)


def test_annotate_factor_both_signs(mock_adata, mock_llm):
    """Test factor annotation with both positive and negative signs."""
    annotate_factor(mock_adata, "pca", mock_llm, sign="both", top_genes=3)

    # Check that we have both positive and negative signs for each factor
    signs_per_factor = (
        pd.DataFrame(mock_adata.uns["factor_annotation"]["raw"])
        .groupby("factor")["sign"]
        .nunique()
    )
    assert all(signs_per_factor == 2)  # Both + and - signs


def test_annotate_factor_invalid_varm_key(mock_adata, mock_llm):
    """Test factor annotation with invalid varm key."""
    with pytest.raises(KeyError):
        annotate_factor(mock_adata, "invalid_key", mock_llm)


def test_annotate_factor_empty_factors(mock_adata, mock_llm):
    """Test factor annotation with empty factors list."""
    with pytest.raises(ValueError):
        annotate_factor(mock_adata, "pca", mock_llm, factors=[])


def test_annotate_factor_invalid_sign(mock_adata, mock_llm):
    """Test factor annotation with invalid sign."""
    with pytest.raises(ValueError):
        annotate_factor(mock_adata, "pca", mock_llm, sign="invalid")


def test_annotate_factor_zero_top_genes(mock_adata, mock_llm):
    """Test factor annotation with zero top genes."""

    with pytest.raises(ValueError):
        annotate_factor(mock_adata, "pca", mock_llm, top_genes=0)


def test_annotate_factor_large_top_genes(mock_adata, mock_llm):
    """Test factor annotation with top_genes larger than available genes."""
    with pytest.raises(ValueError):
        annotate_factor(mock_adata, "pca", mock_llm, top_genes=10)
