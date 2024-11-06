from unittest import TestCase

import qsar_readiness
from qsar_readiness import filter_by_size

CHARGED_METALS = ['N#C[Au-]C#N', 'C1C=CC=C1[Ti+2]C1CC=CC=1]']
INTERMETAL = ['Cl[Bi](Cl)(C1=CC=CC=C1)(C1=CC=CC=C1)C1=CC=CC=C1', 'CCCC[Sn]1(CCCC)OC(=O)C=CC(=O)O1',
              'Cl[V](Cl)(C1C=CC=C1)C1C=CC=C1', 'Cl[Zr](Cl)(C1C=CC=C1)C1CC=CC=1']
VALID_WEIRD = ['C1OC2=CC(CCNC3=NC=NC4NN=CC=43)=CC=C2O1',
               'C1SC=CC=1C1=NC(=C2C=CC=CC2=N1)N1C=NC=C1',
               'C=C(Cl)CN1N=C(OC1=O)C1C=CC(F)=CC=1']
VALID_SHORT = ['CCO', 'COCC', 'NCOC(Cl)', '(Cl)ClC=C(Cl)Cl', 'c(F)ccn']
SALTS = ['[Na+].COS([O-])(=O)=O', '[Br-].CCCCCCCCCCCCCC[N+]1=CC=CC=C1', '[Cu++].[O-]N1C=CC=CC1=S.[O-]N1C=CC=CC1=S',
         '[Li+].[O-]S(=O)(=O)C(F)(F)F', '[K+].OC1=CC=C(O)C(=C1)S([O-])(=O)=O',
         '[Co+3].[C-]#N.[H]C1(O)[C@H](OP([O-])(=O)O[C@H](C)CNC(=O)CC[C@]2(C)[C@@H](CC(N)=O)[C@@]3([H])[N-]\\C2=C(C)/C2=N/C(=C\\C4=N\\C(=C(C)/C5=N[C@]3(C)[C@@](C)(CC(N)=O)[C@@H]5CCC(N)=O)\\[C@@](C)(CC(N)=O)[C@@H]4CCC(N)=O)/C(C)(C)[C@@H]2CCC(N)=O)[C@@H](CO)O[C@@H]1N1C=NC2=C1C=C(C)C(C)=C2']
IONIC = ['Cl[Rh](Cl)Cl', 'C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1', 'O=[Sm]O[Sm]=O']


class SaltsAndIntermetallicsTest(TestCase):

    def test_intermetals(self):
        intermetal = ['Cl[V](Cl)(C1C=CC=C1)C1C=CC=C1', 'Cl[Zr](Cl)(C1C=CC=C1)C1CC=CC=1', 'CC[Co]CC']
        long_inter, short_inter = filter_by_size(intermetal, carbon_cutoff=5)
        assert long_inter == intermetal[:2], print('Expected {}, got {}'.format(intermetal[:2], long_inter))
        assert short_inter == [intermetal[2]], print('Expected {}, got {}'.format(intermetal[2], short_inter))
        del long_inter, short_inter

    def test_valid_long(self):
        long_valid, short_valid = filter_by_size(VALID_WEIRD, carbon_cutoff=5)
        assert long_valid == VALID_WEIRD, print('Expected {}, got {}'.format(VALID_WEIRD, long_valid))
        assert short_valid == [], print('Expected {}, got {}'.format([], short_valid))
        del long_valid, short_valid

    def test_size_filter(self):
        long_valid, short_valid = filter_by_size(VALID_SHORT, carbon_cutoff=5)
        assert len(long_valid) == 0, print('Expected [], got {}'.format(long_valid))
        assert short_valid == VALID_SHORT, print('Expected {}, got {}'.format(VALID_SHORT, short_valid))
        del long_valid, short_valid

    def test_intermetals_only_test(self):
        MIXED_INTERMETALS = [*INTERMETAL, *VALID_WEIRD]
        salts, intermetals = qsar_readiness.salts_and_intermetallics(MIXED_INTERMETALS, True)
        assert len(salts) == 0, 'Expected no salts found, {} returned!'.format(salts)
        assert len([x for x in intermetals if x not in INTERMETAL]) == 0, 'Incorrectly classified non-intermetallics!'
        assert len([x for x in intermetals if x in INTERMETAL]) == len(
            INTERMETAL), 'Failed to find all intermetallics: {}'.format([x for x in INTERMETAL if x not in intermetals])
        del salts, intermetals

    def test_salt_only(self):
        MIXED_SALTS = [*SALTS, *INTERMETAL, *VALID_WEIRD]
        salts, intermetals = qsar_readiness.salts_and_intermetallics(MIXED_SALTS, False)
        assert len(intermetals) == 0, 'Expected no intermetallics found, {} returned!'.format(intermetals)
        assert len([x for x in salts if x not in SALTS]) == 0, 'Incorrectly classified non-salts!'
        assert len([x for x in salts if x in SALTS]) == len(
            SALTS), 'Failed to find all salts. \nMissed: {}\nFound: {}\n'.format(
            [x for x in SALTS if x not in salts], [x for x in SALTS if x in salts])
        del salts, intermetals
