from abc import ABC, abstractmethod
from functools import partial

import scanpy as sc

from .chains import _description_chain, _term_chain
from .utils import _prepare_cluster_data, _prepare_factor_data
from .validator import _validate_factors, _validate_sign


class BaseModel(ABC):

    @abstractmethod
    def _get_prompt(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_parser(self):
        raise NotImplementedError()


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

    def fit(self, adata, llm):
        num_factors = adata.varm[self.varm_key].shape[1]
        # num_features = adata.shape[1]
        self.sign = _validate_sign(self.sign)
        self.factors = _validate_factors(self.factors, num_factors)

        # prepare the chain
        data = self._prepare(adata)
        chain = self._get_chain()
        self.results_ = chain(llm).invoke(data)

        if self.copy:
            return self.results_

        adata.uns[self.key_added] = self._postprocess()


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

        # save results to anndata
        # mapping = _prepare_mapping(df, "cluster", "term")
        # var_names = _prepare_var_names(df, mapping)

        # adata.obs[key_added] = adata.obs[cluster_key].astype(str).map(mapping)
        # adata.uns[key_added] = {"raw": out, "mapping": mapping, "var_names": var_names}

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
