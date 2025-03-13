import copy


def sort_ordinals(feature_list, start=0, stop=None, step=1):
    # Use negative steps for descending.

    def is_single(s):
        return all(str(n) not in s for n in range(10, 20))

    iter, max_iter = 0, 10
    remaining = copy.deepcopy(feature_list)
    sorted_list = list()
    i = start
    while len(feature_list) > 0:
        if i < 10:
            t = [
                f for f in feature_list if str(i) in f.lower() and is_single(f.lower())
            ]
        elif i >= 10:
            t = [f for f in feature_list if str(i) in f.lower()]
            [remaining.pop(f) for f in t]
            if len(t) > 1 or (len(t) == 1 and type(t) is list):
                sorted_list.extend(t)
            if len(t) == 1:
                sorted_list.append(t)
        i += step
        iter += 1
        if (step > 0 and i > stop) or (step < 0 and i < stop) or (iter > max_iter):
            break
    return sorted_list


def iterate_feature_pca(
    feature_df,
    new_feat,
    previous_subset,
    previous_pca=None,
    evr_thresh=(0.9,),
    delta_thresh=None,
    **pca_kwargs
):
    from sklearn.decomposition import PCA

    if previous_pca is None:
        previous_pca = PCA(**pca_kwargs).fit(feature_df[previous_subset])
    new_pca = PCA(**pca_kwargs).fit(feature_df[previous_subset.append(new_feat)])
    if evr_thresh is not None and evr_thresh[0] is not None:
        t_len = min(len(new_pca.explained_variance_ratio_), len(evr_thresh))
        if not all(
            [new_pca.explained_variance_ratio_[i] < evr_thresh[i] for i in t_len]
        ):
            return False, new_pca
    if delta_thresh is not None and delta_thresh[0] is not None:
        t_len = min(
            len(new_pca.explained_variance_ratio_),
            len(delta_thresh),
            len(new_pca.explained_variance_ratio_),
        )
        if not all(
            [
                (
                    1
                    - new_pca.explained_variance_ratio_[i]
                    / previous_pca.explained_variance_ratio_[i]
                    > delta_thresh[i]
                )
                for i in t_len
            ]
        ):
            return False, new_pca
    return True, new_pca


def thresholded_group_pca(
    feature_df, subset_list, smallest_size=3, evr_thresh=0.925, **pca_kwargs
):
    previous_pca = None
    pca_check, new_pca = iterate_feature_pca(
        feature_df,
        subset_list[smallest_size],
        subset_list[: smallest_size - 1],
        evr_thresh=(evr_thresh,),
        **pca_kwargs
    )
    while pca_check and new_pca.n_features_in_ < len(subset_list):
        previous_pca = new_pca
        pca_check, new_pca = iterate_feature_pca(
            feature_df,
            subset_list[new_pca.n_features_in_],
            subset_list[: new_pca.n_features_in_ + 1],
            previous_pca=previous_pca,
            **pca_kwargs
        )
    return previous_pca
