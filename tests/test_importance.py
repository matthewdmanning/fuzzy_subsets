from unittest import TestCase

from sklearn.datasets import make_classification


class VIFTests(TestCase):
    def test_stochastic_vif(self):
        data, target = make_classification(
            n_samples=5000,
            n_features=100,
            n_classes=2,
            n_clusters_per_class=5,
            n_informative=25,
            n_redundant=75,
            random_state=True,
        )

        # current_data, target = load_breast_cancer(return_X_y=True, as_frame=True)

        self.fail()
