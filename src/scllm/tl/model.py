from typing import Literal

import numpy as np
import pandas as pd

from .base import (
    BaseModel,
    ClusterMixin,
    DescriptionMixin,
    FactorMixin,
    TermMixin,
    TermsMixin,
)
from .prompts import _term_prompt
from .utils import _prepare_mapping, _prepare_var_names

MAPPING = "mapping"
FEATURES = "features"


class ClusterAnnotation(ClusterMixin, TermMixin, BaseModel):
    def __init__(
        self,
        cluster_key: str,
        top_items: int = 30,
        num_samples: int = 1,
        term: str = "cell type",
        feature: str = "genes",
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        epilogue: str | None = None,
        key_added: str = "cluster_annotation",
        copy: bool = False,
    ):
        super().__init__(
            preface=preface, epilogue=epilogue, key_added=key_added, copy=copy
        )
        self.cluster_key = cluster_key
        self.top_items = top_items
        self.num_samples = num_samples

        # prompot related
        self.term = term
        self.feature = feature

    def _get_prompt(self):
        prologue = (
            f"Given the following {self.feature} identify the most likely {self.term}."
        )
        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _postprocess(self, adata):
        df = pd.DataFrame(self.results_)
        mapping = _prepare_mapping(df, "group", "term")
        var_names = _prepare_var_names(df, mapping)

        adata.obs[self.key_added] = adata.obs[self.cluster_key].astype(str).map(mapping)
        adata.uns[self.key_added] = {MAPPING: mapping, FEATURES: var_names}


class ClusterTerms(ClusterMixin, TermsMixin, BaseModel):
    def __init__(
        self,
        cluster_key: str,
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        term: str = "biological processes",
        feature: str = "gene",
        epilogue: str | None = None,
        top_items: int = 30,
        num_samples: int = 1,
        num_terms: int = 5,
        key_added: str = "cluster_terms",
        copy: bool = False,
    ):
        super().__init__(
            preface=preface, epilogue=epilogue, key_added=key_added, copy=copy
        )
        self.cluster_key = cluster_key
        self.term = term
        self.feature = feature
        self.top_items = top_items
        self.num_samples = num_samples
        self.num_terms = num_terms

    def _get_prompt(self):
        prologue = f"Given the following {self.feature} identify the {self.num_terms} most likely {self.term}."

        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _postprocess(self, adata):
        # term mapping
        mapping = {item["group"]: item["term"] for item in self.results_}
        # extract features for each term
        features = {}
        for item in self.results_:
            term_dict = {}
            for i, term in enumerate(item["term"]):
                # double check that the features were really in the data
                union = list(set(item["features"][i]) & set(item["data"]))
                term_dict[term] = union

            features[item["group"]] = term_dict

        adata.uns[self.key_added] = {MAPPING: mapping, FEATURES: features}
        # pass


class ClusterDescription(ClusterMixin, DescriptionMixin, BaseModel):
    def __init__(
        self,
        cluster_key: str,
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        feature: str = "genes",
        epilogue: (
            str | None
        ) = "What could kind of cell type could these genes refer to ?",
        top_items: int = 30,
        key_added: str = "cluster_description",
        copy: bool = False,
    ):
        super().__init__(
            preface=preface, epilogue=epilogue, key_added=key_added, copy=copy
        )
        self.cluster_key = cluster_key
        self.top_items = top_items
        self.num_samples = 1

        # prompt related
        self.feature = feature

    def _get_prompt(self):
        prologue = f"You are given the following {self.feature}."

        return _term_prompt(
            preface=self.preface,
            prologue=prologue,
            epilogue=self.epilogue,
            format=False,
        )

    def _postprocess(self, adata):
        mapping = {item["group"]: np.str_(item["target"]) for item in self.results_}
        adata.uns[self.key_added] = {MAPPING: mapping}


class FactorAnnotation(TermMixin, FactorMixin, BaseModel):
    def __init__(
        self,
        varm_key: str,
        factors: list[str] | str = "all",
        sign: Literal["+", "both"] = "both",
        top_items: int = 30,
        num_samples: int = 1,
        term: str = "cell type",
        feature: str = "gene",
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        epilogue: str | None = None,
        key_added: str = "factor_annotation",
        copy: bool = False,
    ):
        super().__init__(
            preface=preface, epilogue=epilogue, key_added=key_added, copy=copy
        )
        self.varm_key = varm_key
        self.factors = factors
        self.sign = sign
        self.top_items = top_items
        self.num_samples = num_samples

        # prompt params
        self.term = term
        self.feature = feature

    def _get_prompt(self):
        prologue = (
            f"Given the following {self.feature} identify the most likely {self.term}."
        )
        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _postprocess(self, adata):
        df = pd.DataFrame(self.results_).assign(
            id=lambda df: df.apply(lambda row: row.factor + row.sign, 1)
        )
        mapping = _prepare_mapping(df, "id", "term")
        var_names = _prepare_var_names(df, mapping)

        adata.uns[self.key_added] = {MAPPING: mapping, FEATURES: var_names}


class FactorTerms(TermsMixin, FactorMixin, BaseModel):
    def __init__(
        self,
        varm_key: str,
        factors: list[str] | str = "all",
        sign: Literal["+", "both"] = "both",
        top_items: int = 30,
        num_terms: int = 5,
        num_samples: int = 1,
        term: str = "biological processes",
        feature: str = "genes",
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        epilogue: str | None = None,
        key_added: str = "factor_terms",
        copy: bool = False,
    ):
        super().__init__(
            preface=preface, epilogue=epilogue, key_added=key_added, copy=copy
        )
        self.varm_key = varm_key
        self.factors = factors
        self.sign = sign
        self.top_items = top_items
        self.num_terms = num_terms
        self.num_samples = num_samples

        # prompt params
        self.term = term
        self.feature = feature

    def _get_prompt(self):
        prologue = f"Given the following {self.feature} identify the {self.num_terms} most likely {self.term}."

        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _postprocess(self, adata):
        # term mapping
        mapping = {
            item["factor"] + item["sign"]: item["term"] for item in self.results_
        }
        # extract features for each term
        features = {}
        for item in self.results_:
            term_dict = {}
            for i, term in enumerate(item["term"]):
                # double check that the features were really in the data
                union = list(set(item["features"][i]) & set(item["data"]))
                term_dict[term] = union

            features[item["factor"] + item["sign"]] = term_dict

        adata.uns[self.key_added] = {MAPPING: mapping, FEATURES: features}


class FactorDescription(FactorMixin, DescriptionMixin, BaseModel):
    def __init__(
        self,
        varm_key: str,
        factors: list[str] | str = "all",
        sign: Literal["+", "both"] = "both",
        top_items: int = 30,
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        feature: str = "genes",
        epilogue: (
            str | None
        ) = "What could kind of cell type could these genes refer to ?",
        key_added: str = "factor_description",
        copy: bool = False,
    ):
        super().__init__(
            preface=preface, epilogue=epilogue, key_added=key_added, copy=copy
        )
        self.varm_key = varm_key
        self.factors = factors
        self.sign = sign
        self.top_items = top_items
        self.feature = feature
        self.num_samples = 1

    def _get_prompt(self):
        prologue = f"You are given the following {self.feature}."

        return _term_prompt(
            preface=self.preface,
            prologue=prologue,
            epilogue=self.epilogue,
            format=False,
        )

    def _postprocess(self, adata):
        mapping = {
            item["factor"] + item["sign"]: np.str_(item["target"])
            for item in self.results_
        }
        adata.uns[self.key_added] = {MAPPING: mapping}
