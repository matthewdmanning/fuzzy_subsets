import itertools
import logging

import cachetools
import numpy as np
import pandas as pd

import distributions
from constants import names, paths, run_params, selector_params
from features import set_cov_matrix


class FeatureFrame:

    def __init__(self, frame_data, options=None, logger=None, cov_mat=None, **kwargs):
        self._logger = logger
        # self._options = options
        self.paths = paths
        self.names = names
        self.run_params = run_params
        self.selector_params = selector_params
        self._original = frame_data
        self._feat_frame = self._original.copy(deep=True)
        self._feat_vars = None
        self._stats = dict()
        self._nonzero_vars = None
        self._zero_vars = None
        self._sparse = None
        self._dense = None
        self._discrete = None
        self._cont = None
        self._neg = None
        self._nonneg = None
        self._feat_stat_scorer = None
        self._cov_mat = cov_mat
        self._cov_pairs = None
        self._desc_groups = None
        self.fit()

    @property
    def feat_frame(self):
        return self._feat_frame

    """    
    @property
    def original(self):
        return self._original

    @property
    def options(self):
        return self._options
    """

    def fit(self):
        self._set_feat_vars(self.feat_frame)
        self._set_sparse_dense()
        self._set_discrete_cont()
        self._set_neg_nonneg()
        # if self.cov_mat is None:
        #    self._cov_mat = set_cov_matrix(self.feat_frame, self.options.sample_wts)
        # self._cov_pairs = self.set_cov_pairs()
        # self.scalers()
        # feat_types_clf_scorers(None)

    @property
    def feat_vars(self):
        return self._feat_vars

    @property
    def stats(self):
        stats_dict = {"mean", "var", "std", "range", "median", "skew", ""}
        self._stats[""]
        return self._feat_vars

    @feat_vars.setter
    def feat_vars(self, value):
        self._feat_vars = value

    def _set_feat_vars(
        self, df, skipna=True, numeric_only=False, weighted="balanced", *args, **kwargs
    ):
        self._feat_vars = np.var(df.astype(np.float64), ddof=1.0, axis=0)
        self._zero_vars = self._feat_vars[self._feat_vars == 0.0].index
        self._nonzero_vars = self._feat_vars[self._feat_vars > 0.0].index
        self._logger.info("Zero variance features: {}".format(self._zero_vars.size))
        self._logger.info(
            "Non-zero variance features: {}".format(self._nonzero_vars.size)
        )

    @property
    def desc_groups(self):
        return self._desc_groups

    @property
    def feat_stat_scorer(self):
        return self._feat_stat_scorer

    @property
    def discrete(self):
        return self._discrete

    @discrete.setter
    def discrete(self, value):
        if type(value) is not pd.Index:
            raise AttributeError
        self._discrete = value

    @property
    def cont(self):
        return self._cont

    @cont.setter
    def cont(self, value):
        if type(value) is not pd.Index:
            raise AttributeError
        self._cont = value

    def _set_neg_nonneg(self):
        nonneg_cols, neg_cols = list(), list()
        for col, ser in self.feat_frame.items():
            if (ser.values < 0).any():
                neg_cols.append(col)
            else:
                nonneg_cols.append(col)
        self.nonneg = pd.Index(nonneg_cols)
        self.neg = pd.Index(neg_cols)
        self._logger.info("Nonnegative features: {}".format(self.nonneg.size))
        self._logger.info("Features with negative values: {}".format(self.neg.size))

    @cachetools.cached(cache={})
    def _set_sparse_dense(self, all_freq=True, ignore_nan=False):
        if 0 < self.selector_params.tol_sparse < 1:
            freq_cut = 1 - self.selector_params.tol_sparse
        elif 1 < self.selector_params.tol_sparse < self.feat_frame.vary.shape[0]:
            freq_cut = self.selector_params.tol_sparse / self.feat_frame.vary.shape[0]
        else:
            raise ValueError
        # self._logger.info('Sparse tolerance: {}'.format(self.selector_params.tol_sparse))
        # self._logger.info('Freq cut: {}'.format(freq_cut))
        sparse_list, dense_list, zero_list, freq_dict = list(), list(), list(), dict()
        for col, ser in self._pd_feat_freq(ignore_nan=ignore_nan):
            if all_freq:
                freq_dict[col] = ser
            sermax = ser.max(skipna=ignore_nan)
            # print('Feature freq max: {}'.format(sermax))
            logging.debug("Feature maximum: {}".format(sermax))
            if freq_cut < sermax < 1.0:
                sparse_list.append(col)
            elif freq_cut >= sermax:
                dense_list.append(col)
            else:
                zero_list.append(col)
        self._sparse = pd.Index(sparse_list)
        self._dense = pd.Index(dense_list)
        self._logger.info("Sparse features: {}".format(self.sparse.size))
        self._logger.info("Dense features: {}".format(self.dense.size))
        if not run_params.debug:
            if len(self.sparse) < self.selector_params.min_sparse:
                self._logger.warning(
                    "Too few sparse features: \n{}".format(self.sparse)
                )
                raise ValueError
            if len(self.dense) < self.selector_params.min_dense:
                self._logger.warning("Too few dense features: \n{}".format(self.dense))
                raise ValueError

    # TODO: Change to also check for whole-ness and amount of continuity.
    def _set_discrete_cont(self):
        # remainder = self.feat_frame.copy().round(0).subtract(self.feat_frame).astype(float)
        # self._logger.info(remainder.max(axis='rows').sort_values(ascending=False)[:5])
        # self._logger.info(remainder.max(axis='rows').sort_values(ascending=True)[:5])
        count_cols = pd.Index(
            [
                x
                for x in self.feat_frame.columns
                if ("count" in x.lower() or "number" in x.lower())
                and "measure" not in x.lower()
            ]
        )
        non_counts = self.feat_frame.columns.difference(count_cols)
        uniques = self.feat_frame[non_counts].nunique(axis="columns")
        self._logger.info(
            'Columns containing "count" or "number": \n{}'.format(count_cols.size)
        )
        self._discrete = distributions.is_discrete(self.feat_frame)
        # self.discrete = uniques[uniques < (self.selector_params.discrete_thresh * self.feat_frame[non_counts].shape[1])].index.union(count_cols)
        # self.discrete = remainder.columns[((remainder <= self.selector_params.tol_discrete) | (remainder >= (1 - self.selector_params.tol_discrete))).astype(int).sum() == 0]
        self.cont = self.feat_frame.columns.difference(self.discrete)
        self._logger.info("Discrete features: {}".format(self.discrete.size))
        self._logger.info("Continuous features: {}".format(self.cont.size))
        if (
            self.discrete.size < self.selector_params.min_discrete
            or self.cont.size < self.selector_params.min_continuous
        ):
            self._logger.error(
                "Too few features for tol_sparse: {}.".format(
                    self.selector_params.tol_discrete
                )
            )
            raise AttributeError

    @property
    def cov_mat(self):
        return self._cov_mat

    # TODO: Use aweights to find balanced convariance matrix.
    # TODO: Add corelation matrix property.
    def _set_cov_mat(self, val):
        if (
            self.cov_mat is not None
            and type(self.cov_mat) is pd.DataFrame
            and not self.cov_mat.empty
        ):
            self._logger.warning("Covariance matrix has already been calculated.")
        else:
            if type(val) is pd.DataFrame and val.shape == (
                self.dense.size,
                self.dense.size,
            ):
                self._cov_mat = val
            elif type(val) is pd.DataFrame and val.shape == (
                self.feat_frame.columns.size,
                self.feat_frame.columns.size,
            ):
                self._cov_mat = val
            else:
                self._logger.warning(
                    "Covariance matrix is not a DataFrame or is not the correst size. Recalculating..."
                )
                self._cov_mat = set_cov_matrix(
                    self.feat_frame[self.dense],
                    corr_method=self.selector_params.cov_method,
                    sample_wts=self.options.sample_wts,
                )

    @property
    def zero_vars(self):
        return self._zero_vars

    @property
    def nonzero_vars(self):
        return self._nonzero_vars

    @property
    def sparse(self):
        return self._sparse

    @property
    def dense(self):
        return self._dense

    @property
    def cov_pairs(self):
        return self._cov_pairs

    # TODO Refactor this to FeatureSelector bc cov_thresh is independent of current_data.
    def set_cov_pairs(self):
        if type(self._cov_pairs) is list and len(self._cov_pairs) > 0:
            return self._cov_pairs
        if type(self._cov_mat) is not pd.DataFrame:
            cov_mat = pd.DataFrame(
                self._cov_mat,
                index=self._original.columns,
                columns=self._original.columns,
            )
        else:
            cov_mat = self._cov_mat
        cov_mat = cov_mat.loc[self._nonzero_vars, self._nonzero_vars]
        """        diag = np.diag(v=cov_mat)
                if np.count_nonzero(diag) < diag.size:
                    print('Zero variance found in diagonal!!!')
                    print(np.where(diag == 0), flush=True)
                    print(diag, flush=True)
                    raise ValueError
                corr_mat = np.divide(np.divide(cov_mat, diag), diag.T)
        mat_max = cov_mat.max(cov_mat)
        if mat_max <= 1:
            print('Normalization worked!!! The max is {}'.format(mat_max))
            exit()
        """
        # corr_dict = cov_mat.to_dict(orient='series')
        same_list, close_list = list(), list()
        for index_ind, col_ind in itertools.combinations(
            list(range(cov_mat.columns.size)), r=2
        ):
            val = cov_mat.iloc[index_ind, col_ind]
            ind, col = cov_mat.columns[index_ind], cov_mat.columns[col_ind]
            # self._logger(ind, col, val, flush=True)
            if not np.isscalar(val):
                self._logger.warning(
                    "CoV element is not a scalar value: {:.60s}".format(val)
                )
            if np.abs(val) == 1.0:
                same_list.append((col, ind))
                self._logger.warning(
                    "Perfectly correlated feature pair with R of {0:.4f}:\n{1:.30s}\n{2:.30s} ".format(
                        val, ind, col
                    )
                )
            elif np.abs(val) > self.selector_params.cov_thresh:
                # self._logger.info('Highly correlated features with an R of {.25s}\n{.25s}\n{}'.format(val, ind, col))
                close_list.append([(ind, col), val])
        close_list.sort(key=lambda x: x[1], reverse=True)
        self._logger.info("Highly correlated features:")
        for feats, corr in close_list:
            self._logger.info(
                "{0:.4f}: {1:.35s} and {2:.35s}".format(corr, feats[0], feats[1])
            )
        self._cov_pairs = close_list
        return self._cov_pairs

    @cachetools.cached(cache={})
    def _pd_feat_freq(self, ignore_nan=False):
        # self._logger.info('Feature variances: ', self.feat_vars)
        for col, ser in self._original.items():
            yield col, ser.value_counts(normalize=True, dropna=ignore_nan)

    """
    @functools.singledispatch
    def format_options(self, arg):
        self._options = arg

    @format_options.register(dict)
    def _(self, arg):
        for o in REQUIRED_OPTS:
            if o not in arg.keys():
                # self._logger.error('Required option {} not found.'.format(o))
                raise KeyError
        self._options = arg
    """
