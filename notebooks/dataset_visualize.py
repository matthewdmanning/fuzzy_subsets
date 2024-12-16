import itertools

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


def dueling_violins(feature_df, labels, scale=False):
    feature_df = feature_df.copy().dropna()[feature_df.nunique(axis=1) > 1]
    if scale:
        feature_df = StandardScaler().set_output(transform="pandas").fit_transform(feature_df)
    # Highly kurtotic distributions mess up violin plots.
    # kurt = feature_df.var(axis=0).divide(feature_df.kurtosis()).sort_values()
    kurt = feature_df.range(axis=0).sort_values()
    # feature_df = feature_df[feature_df.columns[(feature_df.var(axis=0) > 0.25) & (feature_df.var(axis=0) > 0.25)]]
    # feature_df.reset_index(names="INCHI", inplace=True)
    for colset in itertools.batched(kurt.index, n=4):
        violin_df = feature_df[list(colset)]
        violin_df["Solubility"] = labels
        violin_df.reset_index(drop=True, inplace=True)
        violin_df = violin_df.melt(id_vars=["Solubility"], value_vars=colset, var_name="Feature", ignore_index=True)
        # pprint.pp(violin_df.drop(columns=["Solubility"]))
        # pd.DataFrame.melt()
        # print(violin_df.head())
        # print(violin_df["Feature"].unique())
        # print(violin_df[violin_df.isna()])
        vp = sns.violinplot(
            data=violin_df, x="Feature", y="value", hue="Solubility", dodge=True, bw_adjust=0.75, gap=0.025, inner="quart", split=True
        )
        vp.figure.dpi = 600
        plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
