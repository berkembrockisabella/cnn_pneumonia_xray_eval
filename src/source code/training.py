# =============================================================================
#  training.py — K-Fold Cross-Validation + Fine-Tuning
#  Architectures: ResNet50V2 | DenseNet121 | EfficientNetB0
# =============================================================================

import os, json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50V2, DenseNet121, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report)

from preprocessing import DataPipeline

# Config

SEED          = 42
TARGET_SIZE   = (224, 224)
BATCH_SIZE    = 32
MODELS_DIR    = 'outputs/models'
LOGS_DIR      = 'outputs/logs'

EPOCHS_P1     = 10    # head-only phase
EPOCHS_P2     = 20    # fine-tuning phase
UNFREEZE_LAST = 30    # backbone layers to unfreeze in phase 2
LR_P1         = 1e-3
LR_P2         = 1e-5

LABEL_MAP     = {'NORMAL': 0, 'PNEUMONIA': 1}
CLASS_NAMES   = ['NORMAL', 'PNEUMONIA']
ARCHITECTURES = ['resnet50v2' , 'densenet121', 'efficientnetb0']

tf.random.set_seed(SEED)
np.random.seed(SEED)

for d in [MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)


# Model

def build_model(arch: str, freeze_base: bool = True) -> tuple:
    """
    Builds and returns the full model and a direct reference to the backbone.

    Returns
    -------
    model : tf.keras.Model  — full model (backbone + custom head)
    base  : tf.keras.Model  — backbone only (used later for unfreezing)
    """
    kwargs = dict(include_top=False, weights='imagenet', input_shape=(*TARGET_SIZE, 3))
    bases  = {'resnet50v2': ResNet50V2, 'densenet121': DenseNet121, 'efficientnetb0': EfficientNetB0}

    if arch not in bases:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: {list(bases)}")

    base           = bases[arch](**kwargs)
    base.trainable = not freeze_base

    x   = GlobalAveragePooling2D()(base.output)
    x   = BatchNormalization()(x)
    x   = Dense(256, activation='relu')(x)
    x   = Dropout(0.4)(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=out, name=arch)
    return model, base  # return both so base can be used directly in unfreeze


def unfreeze_top_layers(base: tf.keras.Model, n: int = UNFREEZE_LAST) -> None:
    """
    Unfreezes the last `n` layers of the backbone. Modifies `base` in-place.
    BatchNormalization layers are always kept frozen to preserve learned statistics.

    Parameters
    ----------
    base : backbone model returned by build_model
    n    : number of layers (from the end) to unfreeze
    """
    base.trainable = True

    for layer in base.layers[:len(base.layers) - n]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


# Generator

def make_generator(datagen, X, y, shuffle: bool = True):
    return datagen.flow_from_dataframe(
        dataframe   = pd.DataFrame({'filename': X, 'class': y}),
        x_col       = 'filename',
        y_col       = 'class',
        target_size = TARGET_SIZE,
        color_mode  = 'rgb',
        class_mode  = 'binary',
        batch_size  = BATCH_SIZE,
        shuffle     = shuffle,
        seed        = SEED
    )


# Training

def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return {LABEL_MAP[c]: w for c, w in zip(classes, weights)}


def get_callbacks(arch: str, fold: int, phase: int, experiment: str = ''):
    exp_models = f'{MODELS_DIR}/{experiment}' if experiment else MODELS_DIR
    exp_logs   = f'{LOGS_DIR}/{experiment}'   if experiment else LOGS_DIR
    os.makedirs(exp_models, exist_ok=True)
    os.makedirs(exp_logs,   exist_ok=True)
    prefix = f'{exp_models}/{arch}_fold{fold}_phase{phase}'
    return [
        EarlyStopping(monitor='val_loss', patience=5 if phase == 1 else 8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3 if phase == 1 else 5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'{prefix}_best.keras', monitor='val_loss',
                        save_best_only=True, verbose=0),
        CSVLogger(f'{exp_logs}/{arch}_fold{fold}_phase{phase}.csv')
    ]


def train_fold(arch: str, fold_data: dict, train_datagen, val_datagen,
               experiment: str = '') -> dict:
    fold             = fold_data['fold']
    X_train, y_train = fold_data['X_train'], fold_data['y_train']
    X_val,   y_val   = fold_data['X_val'],   fold_data['y_val']

    print(f"\n{'='*55}\n  Fold {fold} | {arch.upper()}\n{'='*55}")

    cw        = get_class_weights(y_train)
    train_gen = make_generator(train_datagen, X_train, y_train, shuffle=True)
    val_gen   = make_generator(val_datagen,   X_val,   y_val,   shuffle=False)
    steps_tr  = max(1, len(X_train) // BATCH_SIZE)
    steps_val = max(1, len(X_val)   // BATCH_SIZE)

    print(f"  Class weights: {cw}")

    def compile_and_fit(model, lr, epochs, phase):
        model.compile(Adam(lr), 'binary_crossentropy',
                      metrics=['accuracy',
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall')])
        return model.fit(train_gen, steps_per_epoch=steps_tr,
                         validation_data=val_gen, validation_steps=steps_val,
                         epochs=epochs, class_weight=cw,
                         callbacks=get_callbacks(arch, fold, phase, experiment), verbose=1)

    print(f"\n  [Phase 1] Head only — LR={LR_P1}")
    model, base = build_model(arch, freeze_base=True)  # keep direct reference to backbone
    hist1 = compile_and_fit(model, LR_P1, EPOCHS_P1, phase=1)

    print(f"\n  [Phase 2] Fine-tuning — LR={LR_P2}")
    unfreeze_top_layers(base)                           # unfreeze via direct reference
    hist2 = compile_and_fit(model, LR_P2, EPOCHS_P2, phase=2)

    return dict(fold=fold, arch=arch, model=model,
                history_p1=hist1.history, history_p2=hist2.history,
                val_gen=val_gen, y_val=y_val)


# Evaluation

def predict(model, gen, y_true, threshold=0.5):
    n_steps    = int(np.ceil(len(y_true) / gen.batch_size))
    gen.reset()
    y_prob     = model.predict(gen, steps=n_steps, verbose=0).flatten()
    y_pred     = (y_prob >= threshold).astype(int)
    y_true_int = np.array([LABEL_MAP[l] for l in y_true])
    return y_true_int, y_pred


def compute_metrics(y_true, y_pred, fold, arch) -> dict:
    return {
        'fold': fold, 'arch': arch,
        'accuracy'          : accuracy_score (y_true, y_pred),
        'precision_macro'   : precision_score(y_true, y_pred, average='macro',    zero_division=0),
        'recall_macro'      : recall_score   (y_true, y_pred, average='macro',    zero_division=0),
        'f1_macro'          : f1_score       (y_true, y_pred, average='macro',    zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted'   : recall_score   (y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted'       : f1_score       (y_true, y_pred, average='weighted', zero_division=0),
    }


def evaluate_fold(result: dict) -> dict:
    y_true, y_pred = predict(result['model'], result['val_gen'], result['y_val'])
    return compute_metrics(y_true, y_pred, result['fold'], result['arch'])


def evaluate_test_set(model, X_test, y_test, val_datagen, arch: str) -> dict:
    test_gen       = make_generator(val_datagen, X_test, y_test, shuffle=False)
    y_true, y_pred = predict(model, test_gen, y_test)
    metrics        = compute_metrics(y_true, y_pred, fold=-1, arch=arch)
    metrics['split'] = 'test'
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    return metrics