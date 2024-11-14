###  Author: Octavio Mesner
###  This code should be able to compute
###  Conditional Mutual Information using kNN
import functools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.special import digamma
from sklearn.conftest import set_config

# If a variable is categorical, M/F eg, we must define a distance
# if not equal
# might need to be changed to something else later
set_config(**{"transform_output": pd.DataFrame, "enable_cython_pairwise_dist": 1})


class MutualInformationStation:

    def __init__(self, df):
        self.whole_data = df
        self.current_data = None
        self.distArray = None
        self.discrete_dist = 1
        # encoded_target = LabelEncoder().fit_transform(y=namedata[y[0]])

    @functools.lru_cache(maxsize=10000, typed=False)
    def getPairwiseDistArray(self, x, y, z=None):
        """
         Input:
        self.current_data: pandasself.current_data frame
         coords: list of indices for variables to be used
         discrete_dist: distance to be used for non-numeric differences

         Output:
         p x n x n array with pairwise distances for each variable
        """
        n, p = self.current_data.shape
        coords = x, y, z
        if coords is None:
            coords = list(range(p))
        col_names = self.current_data.columns.tolist()
        distArray = np.empty([p, n, n])
        distArray[:] = np.nan
        for coord in coords:
            thisdtype = self.current_data[col_names[coord]].dtype
            print(thisdtype)
            if pd.api.types.is_numeric_dtype(thisdtype):
                distArray[coord, :, :] = abs(
                    self.current_data[col_names[coord]].to_numpy()
                    - self.current_data[col_names[coord]].to_numpy()[:, None]
                )
            else:
                distArray[coord, :, :] = (
                    1
                    - (
                        self.current_data[col_names[coord]].to_numpy()
                        == self.current_data[col_names[coord]].to_numpy()[:, None]
                    )
                ) * self.discrete_dist
        return distArray

    def getPointCoordDists(self, distArray, ind_i, coords=None):
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

    def countNeighbors(self, coord_dists, rho, coords=None):
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

    def getKnnDist(self, distArray, k):
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

    def cmiPoint(self, point_i, x, y, z, k, distArray):
        """
        input:
        point_i: current observation row index
        [x, y, z]: list of indices
        k: positive integer scalar for k in knn
        distArray: output of getPairwiseDistArray

        output:
        conditional_mi point estimate
        """
        from scipy.spatial.distance import squareform

        n = distArray.shape[1]
        distArray = squareform(pdist(self.current_data.to_numpy(), metric="manhattan"))
        coords_dists = np.transpose(
            distArray[coords, :, ind_i]
        )  # => distArray[ind_i , coords]
        # k_tilde = the number of points whose distance is greater than or equal to the fatherest point.
        # rho = list of distances sorted from smallest
        dists = np.max(distArray, axis=1)
        ordered_dists = np.sort(dists)
        # using k, not k-1, here because this includes dist to self
        k_tilde = (
            np.count_nonzero(dists <= ordered_dists[k]) - 1
        )  # number of points closer than [k+1]th point minus
        # 1. Guaranteed to be k unless dist calculation or sort got FUBARed rho = distance list for kth point
        return k_tilde, ordered_dists[k]

        # coord_dists = self.getPointCoordDists(distArray, point_i, x + y + z)
        # k_tilde, rho = self.getKnnDist(coord_dists, k)
        dists = np.max(coord_dists[:, coords], axis=1)
        count = np.count_nonzero(dists <= rho) - 1

        x_coords = list(range(len(x)))
        y_coords = list(range(len(x), len(x + y)))
        z_coords = list(range(len(x + y), len(x + y + z)))
        nxz = self.countNeighbors(coord_dists, rho, x_coords + z_coords)
        nyz = self.countNeighbors(coord_dists, rho, y_coords + z_coords)
        nz = self.countNeighbors(coord_dists, rho, z_coords)
        xi = digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz)
        return xi

    def miPoint(self, point_i, x, y, k, distArray):
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
        coord_dists = self.getPointCoordDists(distArray, point_i, x + y)
        k_tilde, rho = self.getKnnDist(coord_dists, k)
        x_coords = list(range(len(x)))
        y_coords = list(range(len(x), len(x + y)))
        nx = self.countNeighbors(coord_dists, rho, x_coords)
        ny = self.countNeighbors(coord_dists, rho, y_coords)
        xi = digamma(k_tilde) + digamma(n) - digamma(nx) - digamma(ny)
        return xi

    def get_cmi_pt_est(
        self,
        x1,
        x2,
        metric,
    ):
        pass

    def conditional_mi(
        self, x, y, z, n_neighbors=7, dist_metric="euclidean", minzero=1
    ):
        """
        computes conditional mutual information, I(x,y|z)
        input:
        x: list of indices for x
        y: list of indices for y
        z: list of indices for z
        k: hyper parameter for kNN
        self.current_data: pandasself.dataframe

        output:
        scalar value of I(x,y|z)
        """
        # compute CMI for I(x,y|z) usng k-NN
        data_df, original_names, new_ind = self.rename_cols([x, y, z])
        x, y, z = new_ind
        n, p = data_df.shape
        # convert variable to index if not already
        # assert not np.any(data_df.columns.duplicated())
        cols = [x, y, z]
        """
        recols = list()
        for i, lst in enumerate(cols):
            if pd.api.types.is_list_like(lst) and all([type(elem) == str for elem in lst]) and len(lst) > 0:
                print('One of the columns is given as a label, not an index!!!!', flush=True)
                recols[i] = list(data_df).get_indexer(lst)
        if len(recols) < 3:
            recols = x, y, x
        from sklearn.utils import _safe_indexing
        """
        from sklearn.metrics.pairwise import distance

        distance.pdist()
        # distArray = pdist(self.current_data.to_numpy(), metric='manhattan')
        # self.getPairwiseDistArray(coords=cols)
        print("Original array: {}".format(self.distArray), flush=True)
        if len(z) > 0:
            ptEsts = map(
                lambda obs: self.cmiPoint(obs, x, y, z, n_neighbors, self.distArray),
                range(n),
            )
        else:
            ptEsts = map(
                lambda obs: self.miPoint(obs, x, y, n_neighbors, self.distArray),
                range(n),
            )
        print("Point estimates (original): {}".format(ptEsts))
        # first =self.data_df.iloc[:, [*x, *z]]
        # print(first, flush=True)
        # kng = kneighbors_graph(X=first, n_neighbors=n_neighbors, mode='distance', p=1, include_self='auto')
        # print('Weighted nearest neighbors graph from sklearn.')
        # pprint.pp(kng)
        if minzero == 1:
            return max(sum(ptEsts) / n, default=0)
        elif minzero == 0:
            return sum(ptEsts) / n

    def rename_cols(self, col_groups):
        n, p = self.whole_data.shape
        original_cols = self.whole_data.columns.copy()
        seq_list = [list(a) for a in col_groups]
        all_cols = [item for sublist in seq_list for item in sublist]
        print([c for c in all_cols if c not in original_cols])
        # new_df = pd.DataFrame(np.empty([n, len(all_cols)]))
        original_colname, new_colind, ser_list = list(), list(), list()
        c_i = 0
        for i, fl in enumerate(seq_list):
            print(fl)
            original_colname.append(list())
            new_colind.append(list())
            if type(fl) is not list():
                fl = list(fl)
            for c in fl:
                if c in original_cols:
                    new_ind = original_cols.get_loc(c)
                elif (
                    type(c) is int
                    or np.can_cast(type(c), int, casting="equiv")
                    or type(c) is np.int64
                ) and 0 <= c <= p:
                    new_ind = int(c)
                else:
                    print("Could not find {} {} in self.dataFrame".format(c, type(c)))
                    new_ind = original_cols.get_loc(c)
                original_colname[i].append(c)
                ser_list.append(
                    pd.Series(self.whole_data.iloc[new_ind].copy(), name=c_i)
                )
                # new_df.iloc[:, c_i] =self.data_df[c].copy()
                new_colind[i].append(c_i)
                c_i += 1
        new_df = pd.concat(ser_list, axis=1)
        self.current_data = new_df
        return new_df, original_colname, new_colind


# conditional_mi -> [PairwiseDistArray -> L1 norm of X and Y]
def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ["slength", "swidth", "plength", "pwidth", "class"]
    namedata = pd.read_csv(url, names=names)
    print(namedata.head())
    x = ["slength"]
    y = ["class"]


if __name__ == "__main__":
    main()
