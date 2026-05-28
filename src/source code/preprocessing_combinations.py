"""
preprocessing_combinations.py
-----------------------------------------
Experiments all possible combinations of preprocessing/augmentation
techniques on a stratified subsample of 1000 images per dataset.

Special rule:
    The 'ChestX-ray8' dataset is excluded from brightness and contrast
    augmentation (rotation, flips, and shifts are still applied).

Combinations tested (ImageDataGenerator parameters):
    - horizontal_flip     : True / False
    - rotation_range      : 0 / 15
    - width_shift_range   : 0.0 / 0.05
    - height_shift_range  : 0.0 / 0.05
    - zoom_range          : 0.0 / 0.1
    - brightness_range    : None / [0.85, 1.15]
    - equalization        : None / 'hist' / 'adaptive'

Each combination is evaluated via 3-fold cross-validation on the subsampled set
using the architectures defined in ARCHITECTURES (training.py).

Both training phases are applied per fold:
    Phase 1 — frozen backbone, head-only (EPOCHS_P1 epochs)
    Phase 2 — fine-tuning top N layers   (EPOCHS_P2 epochs)

Metrics reported per fold x architecture:
    accuracy, f1_macro, auc

Results are saved to:
    outputs/preprocessing_experiment/results_per_fold.csv  <- raw, for p-value tests
    outputs/preprocessing_experiment/results.csv           <- mean +- std per combo x model
    outputs/preprocessing_experiment/summary.csv           <- ranked by f1_mean

Usage:
    python preprocessing_combinations.py
"""

import gc
import os
import itertools
import warnings
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight as _compute_cw
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from training import (
    build_model, unfreeze_top_layers,
    SEED, TARGET_SIZE, BATCH_SIZE,
    EPOCHS_P1, EPOCHS_P2, UNFREEZE_LAST, LR_P1, LR_P2,
    ARCHITECTURES,
)
from preprocessing import DataPipeline

warnings.filterwarnings('ignore')


def get_class_weights(y: np.ndarray) -> dict:
    """Versão local: aceita y int (0/1) em vez de strings como training.py espera."""
    classes = np.unique(y)
    weights = _compute_cw('balanced', classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OUTPUT_DIR             = 'outputs/preprocessing_experiment'
CACHE_DIR              = 'outputs/preprocessing_experiment/cache'
SAMPLES_PER_DS         = 1000
N_SPLITS               = 3
BATCH_SIZE             = 16                # sobrescreve training.py — menor para caber na RAM
NO_BRIGHTNESS_DATASETS = {'ChestX-ray8'}   # datasets excluídos de brightness/contraste

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────
# COMBINATION GRID
# ─────────────────────────────────────────────

PARAM_GRID = {
    'horizontal_flip': [False, True],
    'rotation_range' : [0, 15],
    'shift_range'    : [0.0, 0.05],   # aplicado em width e height
    'zoom_range'     : [0.0, 0.1],
    'brightness'     : [False, True], # bloqueado para NO_BRIGHTNESS_DATASETS
    'equalization'   : [None, 'hist', 'adaptive'],
}

ALL_COMBINATIONS = [
    dict(zip(PARAM_GRID.keys(), v))
    for v in itertools.product(*PARAM_GRID.values())
]
print(f"Total de combinações: {len(ALL_COMBINATIONS)}")  # 96


# ─────────────────────────────────────────────
# STRATIFIED SUBSAMPLING
# ─────────────────────────────────────────────

def stratified_subsample(df: pd.DataFrame, n_per_dataset: int) -> pd.DataFrame:
    """1000 amostras por dataset, proporcionais à distribuição de classes original."""
    parts = []
    for ds, group in df.groupby('dataset'):
        fracs  = group['label'].value_counts(normalize=True)
        slices = []
        for cls, frac in fracs.items():
            n = min(round(frac * n_per_dataset), len(group[group['label'] == cls]))
            slices.append(group[group['label'] == cls].sample(n, random_state=SEED))
        ds_df = pd.concat(slices)

        shortage = n_per_dataset - len(ds_df)
        if shortage > 0:
            pool  = group.drop(ds_df.index)
            extra = pool.sample(min(shortage, len(pool)), random_state=SEED)
            ds_df = pd.concat([ds_df, extra])

        parts.append(ds_df)
        print(f"  [{ds}] {len(ds_df)} amostras → {dict(ds_df['label'].value_counts())}")

    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


# ─────────────────────────────────────────────
# LOAD IMAGES (com cache em disco)
# ─────────────────────────────────────────────

def load_arrays(df: pd.DataFrame, equalization, pipeline: DataPipeline):
    """
    Carrega imagens usando DataPipeline.preprocess_image (resize + equalização).
    Resultado cacheado em .npz por tipo de equalização — só gera uma vez.
    Retorna X [0,255] float32, y int32, nb_mask bool.
    """
    key        = equalization or 'none'
    cache_path = os.path.join(CACHE_DIR, f'arrays_{key}.npz')

    if os.path.exists(cache_path):
        print(f"  [cache] Carregando {key}.npz...")
        data = np.load(cache_path)
        return data['X'], data['y'], data['nb_mask']

    print(f"  [cache] Gerando {key}.npz...")
    X, y, nb_mask = [], [], []
    for _, row in df.iterrows():
        try:
            # normalize=False → mantém [0,255]; ImageDataGenerator faz rescale=1/255
            img = pipeline.preprocess_image(row['path'], normalize=False,
                                            equalization=equalization)
            X.append(img)
            y.append(0 if row['label'] == 'NORMAL' else 1)
            nb_mask.append(row['dataset'] in NO_BRIGHTNESS_DATASETS)
        except Exception as e:
            print(f"  [WARN] {row['path']}: {e}")

    X       = np.array(X,       dtype=np.float32)
    y       = np.array(y,       dtype=np.int32)
    nb_mask = np.array(nb_mask, dtype=bool)
    np.savez_compressed(cache_path, X=X, y=y, nb_mask=nb_mask)
    return X, y, nb_mask


# ─────────────────────────────────────────────
# AUGMENTATION-AWARE GENERATOR
# ─────────────────────────────────────────────

class PerSampleAugGenerator:
    """
    Aplica brightness_range apenas nas amostras fora de NO_BRIGHTNESS_DATASETS.
    Todas as outras transformações são aplicadas uniformemente.
    """

    def __init__(self, X, y, no_brightness_mask, combo, batch_size, seed=SEED):
        self.X, self.y  = X, y
        self.nb_mask    = no_brightness_mask
        self.batch_size = batch_size
        self.rng        = np.random.default_rng(seed)

        common = dict(
            rescale            = 1./255,
            horizontal_flip    = combo['horizontal_flip'],
            rotation_range     = combo['rotation_range'],
            width_shift_range  = combo['shift_range'],
            height_shift_range = combo['shift_range'],
            zoom_range         = combo['zoom_range'],
            fill_mode          = 'nearest',
        )
        self.gen_full = ImageDataGenerator(
            **common,
            brightness_range=[0.85, 1.15] if combo['brightness'] else None,
        )
        self.gen_nobr = ImageDataGenerator(**common)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __iter__(self):
        idx = np.arange(len(self.X))
        self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch = idx[start:start + self.batch_size]
            bs    = len(batch)
            X_aug = np.empty((bs, *self.X.shape[1:]), dtype=np.float32)
            for j, i in enumerate(batch):
                dgen     = self.gen_nobr if self.nb_mask[i] else self.gen_full
                X_aug[j] = next(dgen.flow(self.X[i:i+1], batch_size=1))[0]
            yield X_aug, self.y[batch]


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'accuracy': float(np.mean(y_pred == y_true)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'auc'     : float(roc_auc_score(y_true, y_prob)
                          if len(np.unique(y_true)) > 1 else 0.0),
    }


def combo_label(combo: dict) -> str:
    parts = []
    if combo['horizontal_flip']: parts.append('flip')
    if combo['rotation_range']:  parts.append(f"rot{combo['rotation_range']}")
    if combo['shift_range']:     parts.append(f"shift{combo['shift_range']}")
    if combo['zoom_range']:      parts.append(f"zoom{combo['zoom_range']}")
    if combo['brightness']:      parts.append('brightness')
    if combo['equalization']:    parts.append(combo['equalization'])
    return '+'.join(parts) or 'baseline'


# ─────────────────────────────────────────────
# TRAIN FOLD (2 fases)
# ─────────────────────────────────────────────

def train_fold(arch, X_tr, y_tr, nb_tr, X_val, y_val, combo) -> dict:
    cw = get_class_weights(y_tr)

    def make_ds(X, y, nb):
        gen = PerSampleAugGenerator(X, y, nb, combo, BATCH_SIZE)
        return tf.data.Dataset.from_generator(
            lambda: iter(gen),
            output_signature=(
                tf.TensorSpec(shape=(None, *TARGET_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,),                 dtype=tf.int32),
            )
        ).prefetch(tf.data.AUTOTUNE), len(gen)

    val_gen = lambda: ImageDataGenerator(rescale=1./255).flow(
        X_val, y_val, batch_size=BATCH_SIZE, shuffle=False
    )

    def callbacks(pat):
        return [
            EarlyStopping(monitor='val_loss', patience=pat,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=max(2, pat - 1), min_lr=1e-7, verbose=0),
        ]

    # Phase 1 — head only
    model, base = build_model(arch, freeze_base=True)
    model.compile(Adam(LR_P1), 'binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    ds1, steps1 = make_ds(X_tr, y_tr, nb_tr)
    model.fit(ds1, epochs=EPOCHS_P1, steps_per_epoch=steps1,
              validation_data=val_gen(), class_weight=cw,
              callbacks=callbacks(3), verbose=0)

    # Phase 2 — fine-tune top N layers
    unfreeze_top_layers(base, UNFREEZE_LAST)
    model.compile(Adam(LR_P2), 'binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    ds2, steps2 = make_ds(X_tr, y_tr, nb_tr)
    model.fit(ds2, epochs=EPOCHS_P2, steps_per_epoch=steps2,
              validation_data=val_gen(), class_weight=cw,
              callbacks=callbacks(5), verbose=0)

    y_prob  = model.predict(val_gen(), verbose=0).flatten()
    metrics = evaluate(y_val, y_prob)
    del model, base
    tf.keras.backend.clear_session()
    gc.collect()
    return metrics


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def run_experiment(df_sub: pd.DataFrame, pipeline: DataPipeline,
                   per_fold_rows_done=None):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    if per_fold_rows_done is not None and len(per_fold_rows_done):
        per_fold_rows = per_fold_rows_done.to_dict('records')
        results = (per_fold_rows_done
                   .groupby(['combo_id', 'label', 'model',
                              'horizontal_flip', 'rotation_range', 'shift_range',
                              'zoom_range', 'brightness', 'equalization'])
                   .agg(acc_mean=('accuracy', 'mean'), acc_std=('accuracy', 'std'),
                        f1_mean =('f1_macro',  'mean'), f1_std =('f1_macro',  'std'),
                        auc_mean=('auc',       'mean'), auc_std=('auc',       'std'))
                   .reset_index().to_dict('records'))
    else:
        results       = []
        per_fold_rows = []

    done_combos = (set(per_fold_rows_done['combo_id'].unique())
                   if per_fold_rows_done is not None else set())
    if done_combos:
        print(f"  Retomando: {len(done_combos)} combos já concluídos, pulando...")

    total = len(ALL_COMBINATIONS)
    for cidx, combo in enumerate(ALL_COMBINATIONS, 1):
        if cidx in done_combos:
            continue

        lbl = combo_label(combo)
        print(f"\n{'─'*65}\n[{cidx}/{total}] {lbl}\n  {combo}")

        X, y, nb = load_arrays(df_sub, combo['equalization'], pipeline)

        for arch in ARCHITECTURES:
            print(f"\n  ▶ {arch.upper()}")
            fold_metrics = []

            for fidx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
                m        = train_fold(arch,
                                      X[tr_idx], y[tr_idx], nb[tr_idx],
                                      X[val_idx], y[val_idx], combo)
                m['fold'] = fidx
                fold_metrics.append(m)

                per_fold_rows.append({
                    'combo_id'       : cidx,
                    'label'          : lbl,
                    'model'          : arch,
                    'fold'           : fidx,
                    'accuracy'       : m['accuracy'],
                    'f1_macro'       : m['f1_macro'],
                    'auc'            : m['auc'],
                    'horizontal_flip': combo['horizontal_flip'],
                    'rotation_range' : combo['rotation_range'],
                    'shift_range'    : combo['shift_range'],
                    'zoom_range'     : combo['zoom_range'],
                    'brightness'     : combo['brightness'],
                    'equalization'   : combo['equalization'] or 'none',
                })
                print(f"    Fold {fidx}: acc={m['accuracy']:.4f}  "
                      f"f1={m['f1_macro']:.4f}  auc={m['auc']:.4f}")

            accs = [m['accuracy'] for m in fold_metrics]
            f1s  = [m['f1_macro'] for m in fold_metrics]
            aucs = [m['auc']      for m in fold_metrics]
            results.append({
                'combo_id'       : cidx,
                'label'          : lbl,
                'model'          : arch,
                'horizontal_flip': combo['horizontal_flip'],
                'rotation_range' : combo['rotation_range'],
                'shift_range'    : combo['shift_range'],
                'zoom_range'     : combo['zoom_range'],
                'brightness'     : combo['brightness'],
                'equalization'   : combo['equalization'] or 'none',
                'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
                'f1_mean' : float(np.mean(f1s)),  'f1_std' : float(np.std(f1s)),
                'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
            })
            print(f"  → {arch}: f1={results[-1]['f1_mean']:.4f}±{results[-1]['f1_std']:.4f}")

        # Salva incrementalmente após cada combo (permite retomar em caso de crash)
        pd.DataFrame(per_fold_rows).to_csv(f'{OUTPUT_DIR}/results_per_fold.csv', index=False)
        pd.DataFrame(results).to_csv(f'{OUTPUT_DIR}/results.csv', index=False)
        del X, y, nb
        gc.collect()

    return pd.DataFrame(results), pd.DataFrame(per_fold_rows)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices('GPU'))

    # Reutiliza DataPipeline apenas para carregar e limpar metadados
    pipeline = DataPipeline(base_path='datasets', target_size=TARGET_SIZE,
                            graphs_dir='outputs/graphs', random_state=SEED)
    pipeline._load_metadata()
    pipeline._remove_corrupted()
    df_all = pipeline.df[['dataset', 'label', 'path']].copy()

    # Amostragem estratificada — 1000 por dataset
    print(f"\nAmostragem estratificada ({SAMPLES_PER_DS}/dataset):")
    df_sub = stratified_subsample(df_all, SAMPLES_PER_DS)
    print(f"\nTotal subsample: {len(df_sub)}")
    print(df_sub.groupby(['dataset', 'label']).size()
                .reset_index(name='count').to_string(index=False))

    # Retoma de onde parou, se existir arquivo parcial
    per_fold_path      = f'{OUTPUT_DIR}/results_per_fold.csv'
    per_fold_rows_done = pd.read_csv(per_fold_path) if os.path.exists(per_fold_path) else None

    # Experimento
    results_df, per_fold_df = run_experiment(df_sub, pipeline, per_fold_rows_done)

    # Salva resultados finais
    per_fold_df.to_csv(f'{OUTPUT_DIR}/results_per_fold.csv', index=False)
    results_df.to_csv( f'{OUTPUT_DIR}/results.csv',          index=False)

    summary = results_df.sort_values('f1_mean', ascending=False).reset_index(drop=True)
    summary.to_csv(f'{OUTPUT_DIR}/summary.csv', index=False)

    cols = ['model', 'label', 'f1_mean', 'f1_std', 'auc_mean', 'acc_mean']
    print("\n" + "="*70)
    print("  TOP 10 — todos os modelos")
    print("="*70)
    print(summary[cols].head(10).to_string(index=True))

    print("\n" + "="*70)
    print("  TOP 5 POR MODELO")
    print("="*70)
    for arch in ARCHITECTURES:
        print(f"\n  {arch.upper()}")
        print(summary[summary['model'] == arch][cols].head(5).to_string(index=True))

    print(f"\nresults_per_fold : {OUTPUT_DIR}/results_per_fold.csv  "
          f"({len(per_fold_df)} linhas: "
          f"{len(ALL_COMBINATIONS)} combos × {len(ARCHITECTURES)} modelos × {N_SPLITS} folds)")
    print(f"results          : {OUTPUT_DIR}/results.csv")
    print(f"summary          : {OUTPUT_DIR}/summary.csv")