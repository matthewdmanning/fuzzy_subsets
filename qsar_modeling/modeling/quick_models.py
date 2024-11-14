import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer, matthews_corrcoef


def get_predictions(
    model, train_data, train_labels, test_data, sample_wts=None, **model_params
):
    if model_params:
        model.set_params(**model_params)
    model.fit(X=train_data, y=train_labels, sample_weight=sample_wts)
    train_predict = pd.Series(model.predict(X=train_data), index=train_data.index)
    test_predict = pd.Series(model.predict(X=test_data), index=test_data.index)
    train_proba = pd.DataFrame(
        model.predict_proba(X=train_data), index=train_data.index
    )
    test_proba = pd.DataFrame(model.predict_proba(X=test_data), index=test_data.index)
    return model, train_predict, test_predict, train_proba, test_proba


def balanced_forest(
    train_data, train_labels, test_data, sample_wts=None, **model_params
):
    n_trees = 100
    model = BalancedRandomForestClassifier(
        n_estimators=n_trees,
        max_depth=15,
        n_jobs=-1,
        random_state=0,
        sampling_strategy="auto",
        replacement=False,
        verbose=0,
        class_weight="balanced_subsample",
        bootstrap=True,
    )
    return get_predictions(
        model,
        train_data,
        train_labels,
        test_data,
        sample_wts=sample_wts,
        **model_params
    )


# min_weight_fraction_leaf=0.05,


def logistic_clf(train_data, train_labels, test_data, sample_wts=None, **model_params):
    model = LogisticRegressionCV(
        max_iter=500,
        solver="newton-cg",
        class_weight="balanced",
        cv=3,
        n_jobs=-1,
        scoring=make_scorer(matthews_corrcoef),
        random_state=0,
    )
    return get_predictions(model, train_data, train_labels, test_data, **model_params)


def cv_predictions(estimator, feature_df, labels, cv=None, sample_wts=None):
    score_list = list()
    for dev_X, dev_y, eva_X, eva_y in cv_tools.split_df(
        feature_df, labels, splitter=cv
    ):
        model, train_predict, test_predict, train_proba, test_proba = get_predictions(
            estimator, dev_X, dev_y, eva_X, eva_y, sample_wts
        )
        score_dict = cv_tools.score_cv_results(
            cv_tools.package_output(dev_y, eva_y, (train_predict, test_predict))
        )
        score_list.append(score_dict)
    log_score_summary
