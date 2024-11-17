from __future__ import annotations


def get_features_dict(feature_list):
    mol_wt = [c for c in feature_list if "molecular weight" in c.lower()]
    logp = [c for c in feature_list if "logp" in c.lower()]
    rotate = [
        c
        for c in feature_list
        if "rotat" in c.lower() and ("include" in c.lower() and "terminal" in c.lower())
    ]
    fused = [
        c
        for c in feature_list
        if ("rings" in c.lower() and "fused" in c.lower() and "hetero" not in c.lower())
        and "membered" in c.lower()
    ]
    plain_hetero = [
        c
        for c in feature_list
        if ("rings" in c.lower() and "fused" not in c.lower() and "hetero" in c.lower())
        and "membered" in c.lower()
    ]
    all_rings = [
        c
        for c in feature_list
        if "rings" in c.lower() and "membered" in c.lower() and "all" in c.lower()
    ]
    size_rings = [
        c
        for c in feature_list
        if "rings" in c.lower() and "membered" in c.lower() and "all" not in c.lower()
    ]
    molwalk = [
        c
        for c in feature_list
        if "molecular walk count" in c.lower() and "return" not in c.lower()
    ]
    molpath = [
        c
        for c in feature_list
        if "molecular path count" in c.lower() and "return" not in c.lower()
    ]  # .append('Total path count (up to order 10)')
    b_ord = [c for c in feature_list if "Conventional bond order" in c]
    all_val = [c for c in feature_list if "Valence path" in c]
    valpath = [
        c for c in all_val if "simple" not in c.lower() and "average" not in c.lower()
    ]
    simpath = [
        c for c in feature_list if "Simple path" in c and "average" not in c.lower()
    ]
    selfwalk = [c for c in feature_list if "returning" in c and "walk" in c.lower()]
    hbond = [
        c
        for c in feature_list
        if (
            (
                "e-state" in c.lower()
                and "hydrogen bond" in c.lower()
                and not (
                    "minimum" in c.lower()
                    or "maximum" in c.lower()
                    or "sum" in c.lower()
                )
            )
            or ("hbond" in c.lower() or "acceptor" in c.lower() or "donor" in c.lower())
        )
    ]
    func_group = [
        c
        for c in feature_list
        if "e-state" in c.lower() and ("count" in c.lower() or "number" in c.lower())
    ]
    # Dictionary Construction
    names_dict = {
        "Bond_Orders": b_ord,
        "All_Valence_Path": all_val,
        "Valence_Path_Counts": valpath,
        "Molecular_Path_Counts": molpath,
        "Molecular_Walk_Count": molwalk,
        "Simple_Path": simpath,
        "Self_Return": selfwalk,
        "Sized_Rings": size_rings,
        "Total_Rings": all_rings,
        "Rotations": rotate,
        "HBond": hbond,
        "FuncGroup": func_group,
        "Mol_Wt": mol_wt,
        "LogP": logp,
    }
    feat_lists = [
        b_ord,
        valpath,
        molpath,
        molwalk,
        simpath,
        selfwalk,
        size_rings,
        all_rings,
        rotate,
        hbond,
        func_group,
        mol_wt,
        logp,
    ]
    # names_dict = dict(zip(list_names, feat_lists))
    return names_dict


def get_descriptor_list(feature_list, include=None, exclude=None):
    fl = feature_list
    if include is not None:
        fl = [f for f in feature_list if all([s.lower() in f.lower() for s in include])]
    if exclude is not None:
        fl = [f for f in fl if all([s.lower() not in f.lower() for s in exclude])]
    return fl


def get_atom_numbers(feature_list):
    return get_descriptor_list(
        feature_list,
        ["number of", "atoms"],
        ["number of atoms", "molecular", "chain", "system"],
    )


def get_estate_counts(feature_list):
    in_kws = ["count", "atom-type", "e-state"]
    out_kws = ["h e-state"]
    return get_descriptor_list(feature_list, include=in_kws, exclude=out_kws)
