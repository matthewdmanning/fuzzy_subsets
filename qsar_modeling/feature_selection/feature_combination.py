import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from type_check import is_iterable, make_lowercase


def pca_features(
    train_data, test_data, train_labels=None, oversampler=False, **pca_params
):
    if oversampler and train_labels is not None:
        pca_ready_data, pca_ready_labels = RandomOverSampler(
            random_state=0
        ).fit_resample(train_data, train_labels)
    else:
        pca_ready_data = train_data.copy()
    robust = RobustScaler(unit_variance=True).fit(X=train_data)
    feature_pca = PCA(n_components=1, svd_solver="randomized", **pca_params).fit(
        X=robust.transform(pca_ready_data),
    )
    pca_dev_arr = feature_pca.transform(robust.transform(pca_ready_data))
    pca_eva_arr = feature_pca.transform(robust.transform(pca_ready_data))
    pca_dev_X = pd.DataFrame(data=pca_dev_arr, index=train_data.index)
    pca_eva_X = pd.DataFrame(data=pca_eva_arr, index=test_data.index)
    print("Explained Variance Ratio: {}".format(feature_pca.explained_variance_ratio_))
    print("Principal Components: {}".format(feature_pca.components_))
    return pca_dev_X, pca_eva_X, feature_pca


def pca_threshold(
    train_data,
    test_data,
    group_name,
    evr_thresh=0.95,
    evr_tol=None,
    train_labels=None,
    oversampler=False,
    **pca_params
):
    transformed_list = [
        (train_data.columns.tolist()[0]),
    ]
    prev_evr = 1.0
    for feature in train_data.columns[1:]:
        transformed_list.append(feature)
        print(transformed_list)
        print(train_data.shape, train_labels.shape)
        train_pca_data, test_pca_data, pca_former = pca_features(
            train_data[transformed_list],
            test_data[transformed_list],
            train_labels=train_labels,
            oversampler=oversampler,
            **pca_params
        )
        new_evr = pca_former.explained_variance_ratio_
        if (evr_thresh is not None and new_evr < evr_thresh) or (
            evr_tol is not None and new_evr < prev_evr - evr_tol
        ):
            transformed_list.pop()
            if len(transformed_list) == 1:
                transformed_list.pop()
                train_pca_data, test_pca_data, pca_former = None, None, None
            break
        else:
            prev_evr = new_evr
    if len(train_pca_data.shape) > 0 and train_pca_data[1] > 1:
        col_names = [
            "{}_pca_{}".format(group_name, i + 1)
            for i in list(range(train_pca_data.shape[1]))
        ]
    else:
        col_names = ["{}_pca".format(group_name)]
    train_pca_data.columns, test_pca_data.columns = col_names, col_names
    return train_pca_data, test_pca_data, pca_former


def weight_by_feature(feature_df, to_norm, norm_by):
    return feature_df[to_norm].divide(feature_df[norm_by], axis=0)


def feature_partial_match(
    features, include, exclude, include_all=None, exclude_all=None, case_strict=False
):
    if is_iterable(features):
        return [
            feature_partial_match(f, include, exclude, case_strict=case_strict)
            for f in features
        ]
    if not case_strict:
        [
            make_lowercase(p)
            for p in [features, include, exclude, include_all, exclude_all]
        ]
    for p in [
        p
        for p in [include, exclude, include_all, exclude_all]
        if p is not None and is_iterable(p)
    ]:
        pass
