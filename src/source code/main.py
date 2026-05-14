import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import DataPipeline
from training import (
    train_fold, evaluate_fold, evaluate_test_set,
    SEED, TARGET_SIZE, ARCHITECTURES, MODELS_DIR, LOGS_DIR
)

print("TensorFlow:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Preprocessing pipeline (splits, folds, datagens)

pipeline = DataPipeline(
    base_path='datasets', target_size=TARGET_SIZE, graphs_dir='outputs/graphs',
    random_state=SEED, chestxray8_normal_cap=5000, n_splits=5, test_size=0.15
)
pipeline.run()

folds          = pipeline.folds
X_test, y_test = pipeline.X_test, pipeline.y_test
val_datagen    = pipeline.val_test_datagen   # rescale only — shared by both experiments

# Baseline: no augmentation, rescale only (same as val)
baseline_datagen = ImageDataGenerator(rescale=1./255)

# Preprocessed: with augmentation
augmented_datagen = pipeline.train_datagen

# Experiment runner

def run_experiment(experiment: str, train_datagen) -> tuple:
    """
    Runs K-Fold training + test evaluation for all architectures.

    Parameters
    ----------
    experiment    : label used in checkpoints and log filenames
                    ('baseline' or 'preprocessed')
    train_datagen : ImageDataGenerator used for training folds

    Returns
    -------
    cv_metrics, test_metrics : lists of metric dicts
    """
    exp_models_dir = f'{MODELS_DIR}/{experiment}'
    exp_logs_dir   = f'{LOGS_DIR}/{experiment}'
    os.makedirs(exp_models_dir, exist_ok=True)
    os.makedirs(exp_logs_dir,   exist_ok=True)

    cv_metrics, test_metrics = [], []

    for arch in ARCHITECTURES:
        print(f"\n{'#'*55}\n  [{experiment.upper()}] {arch.upper()}\n{'#'*55}")
        arch_cv = []

        for fold_data in folds:
            result  = train_fold(arch, fold_data, train_datagen, val_datagen,
                                 experiment=experiment)
            metrics = evaluate_fold(result)
            arch_cv.append(metrics)
            cv_metrics.append(metrics)

        # Best fold -> test set evaluation
        best      = max(arch_cv, key=lambda m: m['f1_macro'])
        ckpt_path = f"{exp_models_dir}/{arch}_fold{best['fold']}_phase2_best.keras"
        model     = (tf.keras.models.load_model(ckpt_path)
                     if os.path.exists(ckpt_path) else result['model'])

        t = evaluate_test_set(model, X_test, y_test, val_datagen, arch)
        test_metrics.append(t)

    # Save per-experiment results
    pd.DataFrame(cv_metrics).to_csv(f'{exp_logs_dir}/cv_metrics.csv',    index=False)
    pd.DataFrame(test_metrics).to_csv(f'{exp_logs_dir}/test_metrics.csv', index=False)
    json.dump(test_metrics, open(f'{exp_logs_dir}/test_metrics.json', 'w'), indent=2)
    print(f"\n[{experiment}] Results saved to '{exp_logs_dir}/'")

    return cv_metrics, test_metrics

# Run both experiments

print("\n" + "="*55)
print("  EXPERIMENT 1: BASELINE (no augmentation)")
print("="*55)
baseline_cv, baseline_test = run_experiment('baseline', baseline_datagen)

print("\n" + "="*55)
print("  EXPERIMENT 2: PREPROCESSED (with augmentation)")
print("="*55)
preprocessed_cv, preprocessed_test = run_experiment('preprocessed', augmented_datagen)