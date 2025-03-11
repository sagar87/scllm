from abc import ABC, abstractmethod
from functools import partial

from langchain_core.output_parsers.string import StrOutputParser

from .chains import _description_chain, _term_chain
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

    @abstractmethod
    def _get_chain(self):
        raise NotImplementedError()


class ClusterAnnotation(BaseModel):
    def __init__(
        self,
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        term: str = "cell type",
        feature: str = "gene",
        epilogue: str | None = None,
        top_items: int = 30,
        num_samples: int = 1,
    ):
        super().__init__()
        self.preface = preface
        self.term = term
        self.feature = feature
        self.epilogue = epilogue
        self.top_items = top_items
        self.num_samples = num_samples

    def _prepare(self, adata: str, cluster_key: str):
        data = _prepare_cluster_data(
            adata, cluster_key, top_items=self.top_items, num_samples=self.num_samples
        )
        return data

    def _get_prompt(self):
        prologue = (
            f"Given the following {self.feature} identify the most likely {self.term}."
        )
        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _get_parser(self):
        return _term_parser(self.term, self.feature)

    def _get_chain(self):
        prompt = self._get_prompt()
        parser = self._get_parser()

        return partial(_term_chain, prompt=prompt, parser=parser)

    def fit(self, adata, llm, cluster_key):
        data = self._prepare(adata, cluster_key=cluster_key)
        chain = self._get_chain()
        res = chain(llm).invoke(data)
        return res


class ClusterAnnotationTerms(BaseModel):
    def __init__(
        self,
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        term: str = "biological processes",
        feature: str = "gene",
        epilogue: str | None = None,
        top_items: int = 30,
        num_samples: int = 1,
        num_terms: int = 5,
    ):
        super().__init__()
        self.preface = preface
        self.term = term
        self.feature = feature
        self.epilogue = epilogue
        self.top_items = top_items
        self.num_samples = num_samples
        self.num_terms = num_terms

    def _prepare(self, adata: str, cluster_key: str):
        data = _prepare_cluster_data(
            adata, cluster_key, top_items=self.top_items, num_samples=self.num_samples
        )
        return data

    def _get_prompt(self):
        prologue = f"Given the following {self.feature} identify the {self.num_terms} most likely {self.term}."

        return _term_prompt(
            preface=self.preface, prologue=prologue, epilogue=self.epilogue, format=True
        )

    def _get_parser(self):
        return _multiple_term_parser(self.term, self.feature, self.num_terms)

    def _get_chain(self):
        prompt = self._get_prompt()
        parser = self._get_parser()

        return partial(_term_chain, prompt=prompt, parser=parser)

    def fit(self, adata, llm, cluster_key):
        data = self._prepare(adata, cluster_key=cluster_key)
        chain = self._get_chain()
        res = chain(llm).invoke(data)
        return res


class ClusterAnnotationDescription(BaseModel):
    def __init__(
        self,
        preface: str = "You are an expert computational biologist who analysis single-cell RNA-seq data",
        feature: str = "genes",
        epilogue: (
            str | None
        ) = "What could kind of cell type could these genes refer to ?",
        top_items: int = 30,
    ):
        super().__init__()
        self.preface = preface
        self.feature = feature
        self.epilogue = epilogue
        self.top_items = top_items

    def _prepare(self, adata: str, cluster_key: str):
        data = _prepare_cluster_data(
            adata, cluster_key, top_items=self.top_items, num_samples=1
        )
        return data

    def _get_prompt(self):
        prologue = f"You are given the following {self.feature}."

        return _term_prompt(
            preface=self.preface,
            prologue=prologue,
            epilogue=self.epilogue,
            format=False,
        )

    def _get_chain(self):
        prompt = self._get_prompt()
        parser = self._get_parser()

        return partial(_description_chain, prompt=prompt, parser=parser)

    def _get_parser(self):
        return StrOutputParser()

    def fit(self, adata, llm, cluster_key):
        data = self._prepare(adata, cluster_key=cluster_key)
        chain = self._get_chain()
        res = chain(llm).invoke(data)
        return res
