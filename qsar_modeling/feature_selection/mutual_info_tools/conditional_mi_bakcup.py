###  Author: Octavio Mesner
###  This code should be able to compute
###  Conditional Mutual Information using kNN

import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.utils import tosequence


# If a variable is categorical, M/F eg, we must define a distance
# if not equal
# might need to be changed to something else later


def getPairwiseDistArray(data, coords=None, discrete_dist=1):
    """
    Input:
    current_data: pandas current_data frame
    coords: list of indices for variables to be used
    discrete_dist: distance to be used for non-numeric differences

    Output:
    p x n x n array with pairwise distances for each variable
    """
    n, p = data.shape
    if coords is None:
        coords = []
        coords = range(p)
    col_names = data.columns.tolist()
    distArray = np.empty([p, n, n])
    distArray[:] = np.nan
    for coord in coords:
        thisdtype = data[col_names[coord]].dtype
        if pd.api.types.is_numeric_dtype(thisdtype):
            distArray[coord, :, :] = abs(
                data[col_names[coord]].to_numpy()
                - data[col_names[coord]].to_numpy()[:, None]
            )
        else:
            distArray[coord, :, :] = (
                1
                - (
                    data[col_names[coord]].to_numpy()
                    == data[col_names[coord]].to_numpy()[:, None]
                )
            ) * discrete_dist
    return distArray


def getPointCoordDists(distArray, ind_i, coords=None):
    """
    Input:
    ind_i: current observation row index
    distArray: output from getPariwiseDistArray
    coords: list of variable (column) indices

    output: n x p matrix of all distancs for row ind_i
    """
    if coords is None:
        coords = list()
    if not coords:
        coords = range(distArray.shape[0])
    obsDists = np.transpose(distArray[coords, :, ind_i])
    return obsDists


def countNeighbors(coord_dists, rho, coords=None):
    """
    input: list of coordinate distances (output of coordDistList),
    coordinates we want (coords), distance (rho)

    output: scalar integer of number of points within ell infinity radius
    """

    if coords is None:
        coords = list()
    if not coords:
        coords = range(coord_dists.shape[1])
    dists = np.max(coord_dists[:, coords], axis=1)
    count = np.count_nonzero(dists <= rho) - 1
    return count


def getKnnDist(distArray, k):
    """
    input:
    distArray: numpy 2D array of pairwise, coordinate wise distances,
    output from getPairwiseDistArray
    k: nearest neighbor value

    output: (k, distance to knn)
    """

    dists = np.max(distArray, axis=1)
    ordered_dists = np.sort(dists)
    # using k, not k-1, here because this includes dist to self
    k_tilde = np.count_nonzero(dists <= ordered_dists[k]) - 1
    return k_tilde, ordered_dists[k]


def cmiPoint(point_i, x, y, z, k, distArray):
    """
    input:
    point_i: current observation row index
    x, y, z: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    conditional_mi point estimate
    """
    n = distArray.shape[1]
    coord_dists = getPointCoordDists(distArray, point_i, x + y + z)
    k_tilde, rho = getKnnDist(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x + y)))
    z_coords = list(range(len(x + y), len(x + y + z)))
    nxz = countNeighbors(coord_dists, rho, x_coords + z_coords)
    nyz = countNeighbors(coord_dists, rho, y_coords + z_coords)
    nz = countNeighbors(coord_dists, rho, z_coords)
    xi = digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz)
    return xi


def miPoint(point_i, x, y, k, distArray):
    """
    input:
    point_i: current observation row index
    x, y: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    mi point estimate
    """
    n = distArray.shape[1]
    coord_dists = getPointCoordDists(distArray, point_i, x + y)
    k_tilde, rho = getKnnDist(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x + y)))
    nx = countNeighbors(coord_dists, rho, x_coords)
    ny = countNeighbors(coord_dists, rho, y_coords)
    xi = digamma(k_tilde) + digamma(n) - digamma(nx) - digamma(ny)
    return xi


def rename_cols(data_df, x, y, z):
    n, p = data_df.shape
    original_cols = data_df.columns
    seq_list = [tosequence(a) for a in [x, y, z]]
    all = [list().extend(a) for a in seq_list][0]
    print("Column names in a single list:\n {}".format(all))
    new_df = pd.DataFrame(np.empty([n, len(all)]))
    numeric_indices = list()
    c_i = 0
    for i, fl in enumerate(seq_list):
        numeric_indices.append(list())
        for c in fl:
            if c in original_cols:
                new_ind = data_df.columns.get_loc(c)
            elif type(c) is int and c >= 0 and c < p:
                new_ind = c
            else:
                print("Could not find {} in feature DataFrame".format(c))
                raise KeyError
            numeric_indices[i].append(c)
            new_df[:, i] = data_df[:, c].copy()
    return new_df, numeric_indices

    """
    # convert variable to index if not already
    col_names = [x, y, z]
    new_names, ind_list = list(), list()
    num_col = list()
    for i, col_list in enumerate(col_names):
        if type(col_list) is not list():
            new_names.append(list(col_list))
            ind_list.append(list(data_df.columns.get_loc(col_list)))
        else:
            new_names.append(col_list)
            if any([type(c) is str for c in col_list]):
                ind_list.append(data_df.columns.get_loc(c))
        print(i, col_list)
        if type(col_list) is list and len(col_list) > 1:
            if all([type(col_name) is str for col_name in col_list]):
                num_col.append(data_df.columns.get_indexer_for(c) for c in col_list)
            else:
                num_col.append(col_list)
        elif len(col_list) == 1:
            if type(col_list) is str:
                num_col.append(data_df.columns.get_indexer_for(col_list))
            else:
                num_col.append(col_list)
        else:
            if type(col_list) is str:
                num_col.append(list(data_df.columns.get_indexer_for(col_list)))
            else:
                num_col.append(list(col_list))
    x, y, z = num_col
    """


def conditional_mi(data_df, x, y, z, n_neighbors, discrete_dist=1, minzero=1):
    """
    computes conditional mutual information, I(x,y|z)
    input:
    x: list of indices for x
    y: list of indices for y
    z: list of indices for z
    k: hyperparameter for kNN
    current_data: pandas dataframe

    output:
    scalar value of I(x,y|z)
    """
    # compute CMI for I(x,y|z) using k-NN
    data_df, new_ind = rename_cols(data_df, x, y, z)
    x, y, z = new_ind
    n, p = data_df.shape
    distArray = getPairwiseDistArray(data_df, x + y + z, discrete_dist)
    if len(z) > 0:
        ptEsts = map(
            lambda obs: cmiPoint(obs, x, y, z, n_neighbors, distArray), range(n)
        )
    else:
        ptEsts = map(lambda obs: miPoint(obs, x, y, n_neighbors, distArray), range(n))
    if minzero == 1:
        return max(sum(ptEsts) / n, 0)
    elif minzero == 0:
        return sum(ptEsts) / n


def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ["slength", "swidth", "plength", "pwidth", "class"]
    df = pd.read_csv(url, names=names)
    print(conditional_mi(df, ["slength"], ["class"], ["swidth"], 5))
    pass


if __name__ == "__main__":
    main()
