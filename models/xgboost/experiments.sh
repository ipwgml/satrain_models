#!/bin/bash
python train.py compute.toml --dataset-config dataset_xs.toml
python test.py models/xgboost_gmi_on_swath_xs_tabular_n1000_d6_lr0100_sub080_col080_v0.pkl
python train.py compute.toml --dataset-config dataset_s.toml
python test.py models/xgboost_gmi_on_swath_s_tabular_n1000_d6_lr0100_sub080_col080_v0.pkl
python train.py compute.toml --dataset-config dataset_m.toml
python test.py models/xgboost_gmi_on_swath_m_tabular_n1000_d6_lr0100_sub080_col080_v0.pkl
python train.py compute.toml --dataset-config dataset_xl.toml
python test.py models/xgboost_gmi_on_swath_xl_tabular_n1000_d6_lr0100_sub080_col080_v0.pkl
