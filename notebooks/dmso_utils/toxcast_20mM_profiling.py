import os
import pprint

import pandas as pd

from dmso_utils import data_tools


def get_tox_20():
    tox_20_path = (
        "{}epa_internal/Chemical List CHEMINV_dmsoinsolubles-2024-02-29.csv".format(
            os.environ.get("DATA_DIR")
        )
    )
    tox_20_original = pd.read_csv(
        tox_20_path, usecols=["DTXSID", "CASRN", "INCHIKEY", "SMILES", "INCHI STRING"]
    )
    # print(tox_20_original.columns)
    tox_20_original.dropna(axis="rows", inplace=False)
    tox_20_prefix = tox_20_original.map(
        func=lambda x: x.removeprefix(
            "https://comptox.epa.gov/dashboard/chemical/details/"
        ),
        na_action="ignore",
    )
    return tox_20_prefix


def main():
    tox_20_data = get_tox_20()
    epa_data = dict(
        [(k, v) for k, v in data_tools.load_combo_data().items() if "epa" in k]
    )
    # print([v.columns for v in epa_data.values()])
    overlap_tox = dict(
        [
            (k, tox_20_data[tox_20_data["INCHIKEY"].isin(v["FP_INCHI_KEY"])])
            for k, v in epa_data.items()
        ]
    )
    # overlap_epa = dict([(k, v["FP_INCHI_KEY"][v["FP_INCHI_KEY"].isin(tox_20_data["INCHIKEY"])]) for k, v in epa_data.items()])
    overlap_epa = epa_data["epa_sol"][
        epa_data["epa_sol"]["id"].isin(tox_20_data["DTXSID"])
    ]
    if overlap_epa.shape[0] == 0:
        overlap_epa = epa_data["epa_sol"][
            epa_data["epa_sol"]["INCHI_KEY"].isin(tox_20_data["INCHIKEY"])
        ]
        missing_epa = epa_data["epa_in"][
            ~epa_data["epa_in"]["INCHI_KEY"].isin(tox_20_data["INCHIKEY"])
        ]
    else:
        missing_epa = epa_data["epa_sol"][
            ~epa_data["epa_sol"]["id"].isin(tox_20_data["DTXSID"])
        ]
    # missing_epa = epa_data["epa_in"][~epa_data["epa_in"]["INCHI_KEY"].isin(tox_20_data["INCHIKEY"])]
    print("Total compounds in 5 mM lists.")
    [print("{}: {}".format(k, v.shape[0])) for k, v in epa_data.items()]
    print("Total compounds in 20 mM list.")
    print(tox_20_data.shape)
    print("Number of 20 mM compounds in 5 mM solubility lists.")
    [print("{}: {}".format(k, v.shape)) for k, v in overlap_tox.items()]
    print("Number of 5 mM soluble compounds in 20 mM insoluble list.")
    print(overlap_epa.shape)
    pprint.pp(missing_epa["SMILES_QSAR"], compact=True)
    [print(x) for x in missing_epa["INCHI_KEY"]]
    # overlap_epa[["INCHI_KEY", "SMILES_QSAR", "INCHI", "FP_INCHI", "FP_SMILES", "FP_INCHI", "FP_INCHI_KEY"]].to_csv("{}epa_internal/epa_midsol_5-20.csv".format(os.environ.get("DATA_DIR")))


if __name__ == "__main__":
    main()
