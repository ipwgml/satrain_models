#!/bin/bash
#python train.py compute.toml --dataset-config dataset_xs.toml
#python test.py models/random_forest_gmi_on_swath_xs_tabular_n100_dnull_split2_leaf1_featsqrt_v0.pkl
#python train.py compute.toml --dataset-config dataset_s.toml
#python test.py models/random_forest_gmi_on_swath_s_tabular_n100_dnull_split2_leaf1_featsqrt_v0.pkl
python train.py compute.toml --dataset-config dataset_m.toml
python test.py models/random_forest_gmi_on_swath_m_tabular_n100_dnull_split2_leaf1_featsqrt_v0.pkl
python train.py compute.toml --dataset-config dataset_l.toml
python test.py models/random_forest_gmi_on_swath_l_tabular_n100_dnull_split2_leaf1_featsqrt_v0.pkl
