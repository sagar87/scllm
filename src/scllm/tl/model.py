from abc import ABC, abstractmethod
from functools import partial
from typing import Literal

import scanpy as sc
from langchain_core.output_parsers.string import StrOutputParser

from .chains import _description_chain, _term_chain
from .factor_annotation import _prepare_chain_data, _validate_factors, _validate_sign
from .parser import (
    _multiple_term_parser,
    _term_parser,
)
from .prompts import _term_prompt
from .utils import _prepare_cluster_data


class BaseModel(ABC):

    @abstractmethod
    def _get_prompt(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_parser(self):
        raise NotImplementedError()


class FactorMixin:
    def _prepare(self, adata: sc.AnnData):
        data = _prepare_chain_data(
            adata,
            self.varm_key,
            self.factors,
            "+",
            self.top_items,
            self.num_samples,
        )
        if self.sign == "both":
            data_neg = _prepare_chain_data(
                adata,
                self.varm_key,
                self.factors,
                "-",
                self.top_items,
                self.num_samples,
            )
            data += data_neg

        # Ugly workaround for now
        data = [{"data": d_i["genes"], **d_i} for d_i in data]
        return data

    def fit(self, adata, llm):
        num_factors = adata.varm[self.varm_key].shape[1]
        # num_features = adata.shape[1]
        self.sign = _validate_sign(self.sign)
        self.factors = _validate_factors(self.factors, num_factors)

        data = self._prepare(adata)
        chain = self._get_chain()
        res = chain(llm).invoke(data)
        return res


class ClusterMixin:
    """
    Preprares cluster data.
    """

    def _prepare(self, adata: sc.AnnData):
        data = _prepare_cluster_data(
            adata,
            self.cluster_key,
            top_items=self.top_items,
            num_samples=self.num_samples,
        )
        return data

    def fit(self, adata, llm):
        data = self._prepare(adata)
        chain = self._get_chain()
        res = chain(llm).invoke(data)
        return res


class TermMixin:
    """
    Perpares chains for term-like queries.
    """

    def _get_chain(self):
        prompt = self._get_prompt()
        parser = self._get_parser()

        return partial(_term_chain, prompt=prompt, parser=parser)


class DescriptionMixin:

    def _get_chain(self):
        prompt = self._get_prompt()
        parser = self._get_parser()

        return partial(_description_chain, prompt=prompt, parser=parser)


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
    ):
        super().__init__()
        self.cluster_key = cluster_key
        self.top_items = top_items
        self.num_samples = num_samples
        self.term = term
        self.feature = feature
        self.preface = preface
        self.epilogue = epilogue

    def _get_prompt(self):
        prologue = (
            f"Given the following {self.feature} identify the most likely {self.term}."
        )
        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _get_parser(self):
        return _term_parser(self.term, self.feature)


class ClusterAnnotationTerms(BaseModel, ClusterMixin, TermMixin):
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
    ):
        super().__init__()
        self.cluster_key = cluster_key
        self.preface = preface
        self.term = term
        self.feature = feature
        self.epilogue = epilogue
        self.top_items = top_items
        self.num_samples = num_samples
        self.num_terms = num_terms

    def _get_prompt(self):
        prologue = f"Given the following {self.feature} identify the {self.num_terms} most likely {self.term}."

        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _get_parser(self):
        return _multiple_term_parser(self.term, self.feature, self.num_terms)


class ClusterAnnotationDescription(BaseModel, ClusterMixin, DescriptionMixin):
    def __init__(
        self,
        cluster_key: str,
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        feature: str = "genes",
        epilogue: (
            str | None
        ) = "What could kind of cell type could these genes refer to ?",
        top_items: int = 30,
    ):
        super().__init__()
        self.cluster_key = cluster_key
        self.preface = preface
        self.feature = feature
        self.epilogue = epilogue
        self.top_items = top_items
        self.num_samples = 1

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

    def _get_prompt(self):
        prologue = (
            f"Given the following {self.feature} identify the most likely {self.term}."
        )
        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _get_parser(self):
        return _term_parser(self.term, self.feature)


class FactorAnnotationTerms(BaseModel, TermMixin, FactorMixin):
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
    ):
        super().__init__()
        self.varm_key = varm_key
        self.factors = factors
        self.sign = sign
        self.top_items = top_items
        self.feature = feature

        self.preface = preface
        self.epilogue = epilogue
        self.top_items = top_items
        self.num_samples = 1

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
