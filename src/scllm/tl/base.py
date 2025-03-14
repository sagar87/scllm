from abc import ABC, abstractmethod
from functools import partial

import scanpy as sc
from langchain_core.output_parsers.string import StrOutputParser

from .chains import _description_chain, _term_chain, _terms_chain
from .parser import _multiple_term_parser, _term_parser
from .utils import _prepare_cluster_data, _prepare_factor_data
from .validator import _validate_factors, _validate_sign


class BaseModel(ABC):

    def __init__(
        self, preface: str | None, epilogue: str | None, key_added: str, copy: str
    ):
        self.preface = preface
        self.epilogue = epilogue
        self.key_added = key_added
        self.copy = copy

    @abstractmethod
    def _get_prompt(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_parser(self):
        raise NotImplementedError()

    @abstractmethod
    def _validate(self, adata: sc.AnnData):
        raise NotImplementedError()

    def fit(self, adata, llm):
        self._validate(adata)

        # prepare the chain
        data = self._prepare(adata)
        chain = self._get_chain()
        self.results_ = chain(llm).invoke(data)

        if self.copy:
            return self.results_

        self._postprocess(adata)

        return self


class FactorMixin:
    def _prepare(self, adata: sc.AnnData):
        data = _prepare_factor_data(
            adata,
            self.varm_key,
            self.factors,
            "+",
            self.top_items,
            self.num_samples,
        )
        if self.sign == "both":
            data_neg = _prepare_factor_data(
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

    def _validate(self, adata):
        num_factors = adata.varm[self.varm_key].shape[1]
        # num_features = adata.shape[1]
        self.sign = _validate_sign(self.sign)
        self.factors = _validate_factors(self.factors, num_factors)


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

    def _validate(self, adata):
        if self.cluster_key not in adata.obs.columns:
            raise KeyError(f"The key {self.cluster_key} was not found in obs.")


class TermsMixin:
    """
    Perpares chains for term-like queries.
    """

    def _get_chain(self):
        prompt = self._get_prompt()
        parser = self._get_parser()

        return partial(_terms_chain, prompt=prompt, parser=parser)

    def _get_parser(self):
        return _multiple_term_parser(self.term, self.feature, self.num_terms)


class TermMixin:
    """
    Perpares chains for term-like queries.
    """

    def _get_chain(self):
        prompt = self._get_prompt()
        parser = self._get_parser()

        return partial(_term_chain, prompt=prompt, parser=parser)

    def _get_parser(self):
        return _term_parser(self.term, self.feature)


class DescriptionMixin:

    def _get_chain(self):
        prompt = self._get_prompt()
        parser = self._get_parser()

        return partial(_description_chain, prompt=prompt, parser=parser)

    def _get_parser(self):
        return StrOutputParser()
