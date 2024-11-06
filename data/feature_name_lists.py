from __future__ import annotations


def get_features_dict(feature_list):
    mol_wt = [c for c in feature_list if 'molecular weight' in c.lower()]
    logp = [c for c in feature_list if 'logp' in c.lower()]
    rotate = [c for c in feature_list if 'rotat' in c.lower() and ('include' in c.lower() and 'terminal' in c.lower())]
    fused = [c for c in feature_list if
             ('rings' in c.lower() and 'fused' in c.lower() and not 'hetero' in c.lower()) and 'membered' in c.lower()]
    plain_hetero = [c for c in feature_list if (
                'rings' in c.lower() and not 'fused' in c.lower() and 'hetero' in c.lower()) and 'membered' in c.lower()]
    all_rings = [c for c in feature_list if 'rings' in c.lower() and 'membered' in c.lower() and 'all' in c.lower()]
    size_rings = [c for c in feature_list if
                  'rings' in c.lower() and 'membered' in c.lower() and not 'all' in c.lower()]
    molwalk = [c for c in feature_list if 'molecular walk count' in c.lower() and not 'return' in c.lower()]
    molpath = [c for c in feature_list if
               'molecular path count' in c.lower() and not 'return' in c.lower()]  # .append('Total path count (up to order 10)')
    border = [c for c in feature_list if 'Conventional bond order' in c]
    valpath = [c for c in feature_list if
               'Valence path' in c and not 'simple' in c.lower() and not 'average' in c.lower()]
    simpath = [c for c in feature_list if 'Simple path' in c and not 'average' in c.lower()]
    selfwalk = [c for c in feature_list if 'returning' in c and 'walk' in c.lower()]
    hbond = [c for c in feature_list if
             ('e-state' in c.lower() and 'hydrogen bond' in c.lower() and not (
                         'minimum' in c.lower() or 'maximum' in c.lower() or 'sum' in c.lower()))]
    funct_group = [c for c in feature_list if
                   'e-state' in c.lower() and ('count' in c.lower() or 'number' in c.lower())]
    # Dictionary Construction
    list_names = ['Bond_Orders', 'Valence_Path_Counts', 'Molecular_Path_Counts', 'Molecular_Walk_Count', 'Simple_Path',
                  'Self_Return', 'Sized_Rings', 'Total_Rings', 'Rotations',
                  'HBond', 'FuncGroup', 'Mol_Wt', 'LogP']
    feat_lists = [border, valpath, molpath, molwalk, simpath, selfwalk, size_rings, all_rings, rotate, hbond,
                  funct_group, mol_wt, logp]
    names_dict = dict(zip(list_names, feat_lists))
    return names_dict


def get_descriptor_list(feature_list, include=None, exclude=None):
    fl = feature_list
    if include is not None:
        fl = [f for f in feature_list if all([s.lower() in f.lower() for s in include])]
    if exclude is not None:
        fl = [f for f in fl if all([s.lower() not in f.lower() for s in exclude])]
    return fl


def get_atom_numbers(feature_list):
    return get_descriptor_list(feature_list, ['number of', 'atoms'],
                               ['number of atoms', 'molecular', 'chain', 'system'])


def get_estate_counts(feature_list):
    in_kws = ['count', 'atom-type', 'e-state']
    out_kws = ['h e-state']
    return get_descriptor_list(feature_list, include=in_kws, exclude=out_kws)
