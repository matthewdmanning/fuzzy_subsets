from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd

from utils import math_tools


def find_correlation(
    corr, cutoff=0.9, n_drop=None, exact=True, norm_fn=partial(np.linalg.norm, ord=2)
):
    """
    This function is the Python implementation of the R function
    `find_correlation()`.

    Relies on numpy and pandas, so must have them pre-installed.

    It searches through a correlation matrix and returns a list of column names
    to remove to reduce pairwise correlations.

    For the documentation of the R function, see
    https://www.rdocumentation.org/packages/caret/topics/findCorrelation
    and for the source code of `find_correlation()`, see
    https://github.com/topepo/caret/blob/master/pkg/caret/R/findCorrelation.R

    -----------------------------------------------------------------------------

    Parameters:
    -----------
    corr: pandas dataframe.
        A correlation matrix as a pandas dataframe.
    cutoff: float, default: 0.9.
        A numeric value for the pairwise absolute correlation cutoff
    exact: bool, default: None
        A boolean value that determines whether the average correlations be
        recomputed at each step
    -----------------------------------------------------------------------------
    Returns:
    --------
    list of column names
    -----------------------------------------------------------------------------
    Example:
    --------
    R1 = pd.DataFrame({
        'x1': [1.0, 0.86, 0.56, 0.32, 0.85],
        'x2': [0.86, 1.0, 0.01, 0.74, 0.32],
        'x3': [0.56, 0.01, 1.0, 0.65, 0.91],
        'x4': [0.32, 0.74, 0.65, 1.0, 0.36],
        'x5': [0.85, 0.32, 0.91, 0.36, 1.0]
    }, index=['x1', 'x2', 'x3', 'x4', 'x5'])

    find_correlation(R1, cutoff=0.6, exact=False)  # ['x4', 'x5', 'x1', 'x3']
    find_correlation(R1, cutoff=0.6, exact=True)   # ['x1', 'x5', 'x4']
    """

    def _find_correlation_fast(corr_df, col_center, thresh):

        combsAboveCutoff = (
            corr_df.where(lambda x: (np.tril(x) == 0) & (x > thresh)).stack().index
        )

        rowsToCheck = combsAboveCutoff.get_level_values(0)
        colsToCheck = combsAboveCutoff.get_level_values(1)

        msk = col_center[colsToCheck] > col_center[rowsToCheck].values
        delete_dict = pd.unique(np.r_[colsToCheck[msk], rowsToCheck[~msk]]).tolist()

        return delete_dict

    def _find_correlation_exact(corr_df, max_drop, thresh):
        """
        Removed original code to allow custom sorting function to be used.
        corr_sorted = corr_df.loc[(*[col_center.sort_values(ascending=False).index] * 2,)]

        if (corr_sorted.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            corr_sorted = corr_sorted.astype(float)

        corr_sorted.values[(*[np.arange(len(corr_sorted))] * 2,)] = np.Nan
        """

        corr_sorted = corr_df.copy().astype(float)
        col_norms_sorted = np.argsort(norm_fn(corr_sorted.to_numpy(), axis=0))
        corr_sorted = corr_sorted.iloc[col_norms_sorted]
        # corr_sorted = corr_sorted.loc[col_norms_sorted]
        sorted_indices = math_tools.get_max_arr_indices(corr_sorted)
        delete_dict = OrderedDict()
        for i, j in sorted_indices:
            if len(delete_dict) >= np.abs(max_drop):
                break
            elif i in delete_dict or j in delete_dict or i == j:
                continue
            elif corr_sorted.loc[i, j] < thresh:
                print("All pairs above threshold {} eliminated.".format(thresh))
                break
            i_norm = norm_fn(corr_sorted.loc[i])
            j_norm = norm_fn(corr_sorted.loc[j])
            if i_norm > j_norm:
                delete_dict[i] = j_norm
                corr_sorted.loc[i] = corr_sorted[i] = np.nan
            else:
                delete_dict[j] = j_norm
                corr_sorted.loc[j] = corr_sorted[j] = np.nan
        return delete_dict

    """
    if not np.allclose(corr, corr.T, rtol=1e-3, atol=1e-2):
        raise ValueError("Correlation matrix is not symmetric.")
    elif any(corr.columns != corr.index):
        raise ValueError("Columns aren't equal to indices.")
    """
    acorr = corr.abs()
    avg = np.mean(corr, axis=1)
    if n_drop is None:
        n_drop = int(corr.shape[1] / 2)
    if exact or (exact is None and corr.shape[1] < 100):
        print("Initiating exact correlation search")
        return _find_correlation_exact(acorr, n_drop, cutoff)
    else:
        return _find_correlation_fast(acorr, avg, cutoff)
