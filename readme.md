# Hypergraph-Native Message Passing: An Incidence-Centric Learning Paradigm

## Overview

This repo includes the following code:

1. `train.py`, `datasets.py`, `data_utils.py`, `utils.py`, `models/*.py`. They are modified from [ED-HNN](https://github.com/Graph-COM/ED-HNN) for vertex classification tasks.
2. `main.py`, `generate_node_features.py`, `random_walk_hyper.py`. They are our PyTorch re-implementation of [WHATsNET](https://github.com/young917/EdgeDependentNodeLabel) for hypergraph-dependent labelling tasks.
3. `nt2.py` is our implementation of HMP.
4. Scripts to reproduce our experiments include
   1. `reproduce_vertclass.sh` is for Table 2.
   2. `reproduce_edge.sh` is for Table 1.
   3. `reproduce_hyperchain.sh` is for Figure 5.

## Dependency

Please install Python libraries to meet the code dependencies following the instructions of [ED-HNN](https://github.com/Graph-COM/ED-HNN?tab=readme-ov-file#dependency).

## Data Preparation for Vertex Classification

Please follow the instructions of [ED-HNN](https://github.com/Graph-COM/ED-HNN?tab=readme-ov-file#data-preparation) to prepare datasets `raw_data/` for vertex classificaton.

## Data Preparation for Hypergraph-Dependent Labelling

1. Follow [WHATsNET](https://github.com/young917/EdgeDependentNodeLabel) to prepare `dataset/` which includes the raw text files for all datasets
2. Run `python3 generate_node_features.py <dataset>` to generate `nocefeatires_44d.txt` and move it to `dataset/<dataset>`, where `<dataset>` can be DBLP, AMinerAuthor, emailEnron, emailEu, StackOverflowBiology, or StackOverflowPhysics.
