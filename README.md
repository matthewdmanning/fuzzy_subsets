# DMSO Solubility

Develop augmented cheminformatics models for predicting DMSO solubility (and other endpoints). The approach uses fuzzy set membership, as predicted by structural descriptors, to individually weight submodel contribution.
**Note**: This project is under active development. Source code and notebooks can be found under [dev](https://github.com/MatthewManningEPA/dmso_solubility/tree/dev).

## Description

This project grew out of an effort to model the DMSO solubility of industrial chemicals. When out-of-the-box methods did not exceed previously published solubility models, a structure-based augmented model was proprosed. The ensemble is composed of a base model, which weights compounds based on predictability and weak submodels developed to improve the predicted probabilities for some subset of the data. Since chemical descriptors which accurately predict some compounds may be confounding for others, each submodel performs its own feature selection. The result is a co-clustering of descriptor and chemical space. The final prediction is calculated by voting, using the outputs of the ensemble, weighted using fingerprints not present in the feature selection space.

Both code development and analysis of preliminary results are on-going.

## Getting Started

### Dependencies

_Some features (SMILES standardization, featurization, and chemical ID conversion) use internal EPA APIs. These are necessary for the current use involving data from the CompTox Dashboard, but not required to use the modeling and visualization tools._

## Authors

[Matthew Manning](Manning.Matthew@epa.gov)

## License

This project is licensed under the [MIT] License - see the LICENSE.md file for details
