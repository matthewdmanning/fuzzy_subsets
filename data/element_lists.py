ALL_ELEMENTS = list(['H',
                     'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                     'K',
                     'Ca', 'Sc', 'Ti',
                     'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
                     'Y',
                     'Zr', 'Nb', 'Mo',
                     'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
                     'Pr',
                     'Nd', 'Pm', 'Sm',
                     'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                     'Au',
                     'Hg', 'Tl', 'Pb',
                     'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
                     'Es',
                     'Fm', 'Md', 'No',
                     'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'])
NONMETALS = list(['B', 'C', 'N', 'O', 'P', 'S'])
NONMETALS_LOWER = list(['b', 'c', 'n', 'o', 'p', 's'])
ATYPICAL = list(['B', 'Si', 'Se', 'Te'])
HALOGENS = list(['F', 'Cl', 'Br', 'I'])
GROUP_ONES = list(['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'])
GROUP_TWOS = list(['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'])
ALLOWED = [*NONMETALS, *HALOGENS, *NONMETALS_LOWER, 'H']
DISALLOWED = list([e for e in ALL_ELEMENTS if e not in ALLOWED])
QSAR_COLUMNS = {'id': 'FOREIGN_KEY', 'cid': 'CID', 'sid': 'SID', 'casrn': 'CASRN', 'name': 'NAME', 'smiles': 'SMILES_QSAR',
                'canonicalSmiles': 'SMILES_CANONICAL', 'inchi': 'INCHI', 'inchiKey': 'INCHI_KEY', 'mol': 'MOL'}
DESC_COLUMNS = ['DESCRIPTOR_SET', 'VERSION_NUMBER', 'PLACEONE', 'PLACETWO', 'SMILES_QSAR', 'INCHI', 'INCHI_KEY', 'DESCRIPTORS']