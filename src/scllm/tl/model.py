from typing import Literal

import pandas as pd
from langchain_core.output_parsers.string import StrOutputParser

from .base import BaseModel, ClusterMixin, DescriptionMixin, FactorMixin, TermMixin
from .parser import (
    _multiple_term_parser,
    _term_parser,
)
from .prompts import _term_prompt
from .utils import _prepare_mapping


class ClusterAnnotation(BaseModel, ClusterMixin, TermMixin):
    def __init__(
        self,
        cluster_key: str,
        top_items: int = 30,
        num_samples: int = 1,
        term: str = "cell type",
        feature: str = "gene",
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        epilogue: str | None = None,
        key_added: str = "cluster_annotation",
        copy: bool = False,
    ):
        super().__init__()
        self.cluster_key = cluster_key
        self.top_items = top_items
        self.num_samples = num_samples

        # prompot related
        self.term = term
        self.feature = feature
        self.preface = preface
        self.epilogue = epilogue

        # adata related
        self.key_added = key_added
        self.copy = copy

    def _get_prompt(self):
        prologue = (
            f"Given the following {self.feature} identify the most likely {self.term}."
        )
        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _get_parser(self):
        return _term_parser(self.term, self.feature)


class ClusterTerms(BaseModel, ClusterMixin, TermMixin):
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
        super().__init__()
        self.cluster_key = cluster_key
        self.term = term
        self.feature = feature
        self.top_items = top_items
        self.num_samples = num_samples
        self.num_terms = num_terms

        # prompt related
        self.preface = preface
        self.epilogue = epilogue

        # adata related
        self.key_added = key_added
        self.copy = copy

    def _get_prompt(self):
        prologue = f"Given the following {self.feature} identify the {self.num_terms} most likely {self.term}."

        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _get_parser(self):
        return _multiple_term_parser(self.term, self.feature, self.num_terms)


class ClusterDescription(BaseModel, ClusterMixin, DescriptionMixin):
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
        super().__init__()
        self.cluster_key = cluster_key
        self.top_items = top_items
        self.num_samples = 1

        # prompt related
        self.preface = preface
        self.feature = feature
        self.epilogue = epilogue

        # adata related
        self.key_added = key_added
        self.copy = copy

    def _get_prompt(self):
        prologue = f"You are given the following {self.feature}."

        return _term_prompt(
            preface=self.preface,
            prologue=prologue,
            epilogue=self.epilogue,
            format=False,
        )

    def _get_parser(self):
        return StrOutputParser()


class FactorAnnotation(BaseModel, TermMixin, FactorMixin):
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
        super().__init__()
        self.varm_key = varm_key
        self.factors = factors
        self.sign = sign
        self.top_items = top_items
        self.num_samples = num_samples

        # prompt params
        self.term = term
        self.feature = feature
        self.preface = preface
        self.epilogue = epilogue

        # adata related
        self.key_added = key_added
        self.copy = copy

    def _get_prompt(self):
        prologue = (
            f"Given the following {self.feature} identify the most likely {self.term}."
        )
        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _get_parser(self):
        return _term_parser(self.term, self.feature)

    def _postprocess(self):
        df = pd.DataFrame(self.results_).assign(
            id=lambda df: df.apply(lambda row: row.factor + row.sign, 1)
        )
        mapping = _prepare_mapping(df, "id", "term")
        return {"raw": self.results_, "mapping": mapping}


class FactorTerms(BaseModel, TermMixin, FactorMixin):
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
        super().__init__()
        self.varm_key = varm_key
        self.factors = factors
        self.sign = sign
        self.top_items = top_items
        self.num_terms = num_terms
        self.num_samples = num_samples

        # prompt params
        self.term = term
        self.feature = feature
        self.preface = preface
        self.epilogue = epilogue

        # adata related
        self.key_added = key_added
        self.copy = copy

    def _get_prompt(self):
        prologue = f"Given the following {self.feature} identify the {self.num_terms} most likely {self.term}."

        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _get_parser(self):
        return _multiple_term_parser(self.term, self.feature, self.num_terms)


class FactorDescription(BaseModel, FactorMixin, DescriptionMixin):
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
        super().__init__()
        self.varm_key = varm_key
        self.factors = factors
        self.sign = sign
        self.top_items = top_items
        self.feature = feature
        self.num_samples = 1

        self.preface = preface
        self.epilogue = epilogue

        # adata related
        self.key_added = key_added
        self.copy = copy

    def _get_prompt(self):
        prologue = f"You are given the following {self.feature}."

        return _term_prompt(
            preface=self.preface,
            prologue=prologue,
            epilogue=self.epilogue,
            format=False,
        )

    def _get_parser(self):
        return StrOutputParser()
