'''
Outline
Select features that define classes.
Select samples belonging to each minority class.
Select majority subsample and training features.
    -Eliminate *too* sparse features (not enough samples for accurate submodel).
    *Segment current_data based on classification labels. Ensure balanced sampling of target labels.
    -Re-scale features based on predictive performance to ID nearest neighbors.
    -Identify NNs of minority class members in majority class.
    -Sample weights: Use similarity measures between NN and minority class members (additive for multiple NNs).


'''
import logging

import pandas as pd
from sklearn.ensemble import IsolationForest

import distributions

logging.getLogger('dmso_logger.grove_membership')


def filter_too_sparse(features_df, threshold):
    distributions.is_discrete(features_df.columns)
    pass


def classify_samples(features_df, membership_test_list):
    # membership_test_list: List of functions returning True if sample is a member of the class.
    pass


def isolation_grove(features_df, features_sets):
    # Returns list of ndarray of indices of outliers
    outlier_list = list()
    isogrove = IsolationForest(random_state=0, n_jobs=-1)
    for subset in features_sets:
        liers = isogrove.fit_predict(X=features_df[subset])
        outlier_list.append(liers[liers == -1])
    return outlier_list


def get_weighted_features(features_df, feature_names, name_dict=None):
    weight_dict = {'weight': 'Molecular_weight', 'heavy': 'Number of heavy atoms (i.e. not hydrogen)',
                   'carbon': 'Number of carbon atoms'}
    weighted_df = pd.DataFrame(index=features_df.index)
    for key, weighting_name in weight_dict.items():
        new_names = dict(zip(feature_names, [f + ' / {}}'.format(weighting_name) for f in feature_names]))
        for feature, new_name in new_names.items():
            weighted_df[new_name] = features_df[feature].divide(features_df[weighting_name])


def find_syngeristic_features(features_df, target_feature, synergy_matrix):
    # Assume synergy(target) = df[target, candidate]
    bivariate_list = synergy_matrix.loc[target_feature, :]
    bivariate_list.sort_values(ascending=False, inplace=True)


def get_nearest_neighbors(features_df, members):
    pass
