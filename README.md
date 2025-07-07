# TwinC-Manuscript

## Introduction
In this repository, we store scripts to reproduce the analyses presented in [Jha et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11429679/). In summary, you can retrain the TwinC classification model on the heart's left or right ventricle, as well as the TwinC regression model on the H1ESC cell line. TwinC is a tool for training, inference, and interpretation of trans-3D genome folding in humans. TwinC uses a Convolutional Neural Network model that predicts contact between two _trans_ genomic loci from nucleotide sequences. In the classification setting, the model takes two 100-kbp nucleotide sequences as input and predicts the likelihood of contact between them. In the regression setting, the model takes two 640-kbp nucleotide sequences as input and predicts the contact scores for a 5 x 5 patch at 128-kbp resolution. Additionally, we have provided scripts to reproduce the figures in the manuscript in [figure_notebooks directory](https://github.com/Noble-Lab/twinc_paper/tree/main/figure_notebooks).

## [Documentation](https://noble-lab.github.io/twinc/applications/)

## [Citation information](https://noble-lab.github.io/twinc/citing/)
