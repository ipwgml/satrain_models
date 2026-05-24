#!/bin/bash
python train.py compute.toml --dataset-config dataset_geo.toml
python train.py compute.toml --dataset-config dataset_geo_ir.toml
python train.py compute.toml --dataset-config dataset_atms.toml
