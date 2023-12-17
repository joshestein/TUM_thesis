The corresponding code for my master's thesis 'Segmentation of sparse annotated data: application to cardiac imaging'.

Initially, we built a custom pipeline using our own model for our experiments. Corresponding code for training on our custom pipeline can be run from `src/main.py`. Hyperparameters and other variables can be controlled from within `config.toml` in the root directory. Pass the CLI argument `-d` as 'acdc' or 'mnms' depending on the desired dataset. Your data should be setup using the same pre-processing format as nnUNet.

The code for running/training nnUNet can be found in the fork at https://github.com/joshestein/nnUNet/tree/limited_data. The setup/evaluation/inference is similar to the original repo. There are some changes made to include our additional evaluations - see `evaluation/evaluate_all.py`, `evaluation/surface_metrics.py` and `inference/predict_all.py` for some of our important changes.

Code for training our SAM models can be found within `src/sam`. To train models use `src/sam/sam_main.py`. Once trained, inference results can be obtained using `src/sam/sam_inference.py`.
