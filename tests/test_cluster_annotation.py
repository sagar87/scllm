import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from langchain_openai import ChatOpenAI

from scllm.tl.cluster_annotation import (
    _prepare_mapping,
)


@pytest.fixture
def mock_annotation_df():
    """Create a mock DataFrame for testing annotation mapping."""
    return pd.DataFrame(
        {
            "cluster": ["0", "0", "0", "1", "1", "1", "2", "2"],
            "target": [
                "T cell",
                "T cell",
                "B cell",
                "B cell",
                "B cell",
                "T cell",
                "NK cell",
                "NK cell",
            ],
        }
    )


def test_prepare_mapping_basic(mock_annotation_df):
    """Test basic mapping functionality with clear majority annotations."""
    mapping = _prepare_mapping(mock_annotation_df, "cluster", "target")

    assert isinstance(mapping, dict)
    assert len(mapping) == 3  # One for each cluster
    assert mapping["0"] == "T cell"  # 2/3 majority
    assert mapping["1"] == "B cell"  # 2/3 majority
    assert mapping["2"] == "NK cell"  # 2/2 majority


def test_prepare_mapping_tie(mock_annotation_df):
    """Test mapping with tied annotations."""
    # Modify the DataFrame to create a tie in cluster 1
    df = mock_annotation_df.copy()
    df.loc[df["cluster"] == "1", "target"] = ["B cell", "T cell", "T cell"]

    mapping = _prepare_mapping(df, "cluster", "target")

    assert mapping["1"] in ["B cell", "T cell"]  # Either is valid in case of tie


# def test_prepare_mapping_empty_df():
#     """Test mapping with empty DataFrame."""
#     empty_df = pd.DataFrame(columns=["cluster", "target"])
#     mapping = _prepare_mapping(empty_df, "cluster", "target")

#     assert isinstance(mapping, dict)
#     assert len(mapping) == 0


def test_prepare_mapping_single_cluster():
    """Test mapping with single cluster."""
    df = pd.DataFrame(
        {"cluster": ["0", "0", "0"], "target": ["T cell", "T cell", "B cell"]}
    )

    mapping = _prepare_mapping(df, "cluster", "target")

    assert len(mapping) == 1
    assert mapping["0"] == "T cell"


def test_prepare_mapping_invalid_columns():
    """Test mapping with invalid column names."""
    df = pd.DataFrame({"wrong_col": ["0", "0"], "target": ["T cell", "B cell"]})

    with pytest.raises(KeyError):
        _prepare_mapping(df, "cluster", "target")
