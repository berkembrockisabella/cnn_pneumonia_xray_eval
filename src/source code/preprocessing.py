import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataPipeline:
    """
    Encapsulates the EDA and preprocessing steps for the pneumonia dataset.

    Split strategy:
        1. A fixed holdout test set is separated before any CV.
           It never participates in training or validation — used
           exclusively for the final unbiased model evaluation.
        2. The remaining data is split via StratifiedKFold into K folds.
           In each fold, (K-1) parts form the training set and 1 part forms
           the validation set, ensuring every sample is validated exactly once.

    Balancing:
        Only the NORMAL class from ChestX-ray8 is sampled (~60k → configurable cap).
        All other datasets are kept intact.
        Residual class imbalance is handled via class_weight during training,
        without discarding valid samples from other datasets.

    Usage:
        pipeline = DataPipeline(base_path='datasets')
        pipeline.run()
    """

    CLASSES = {'NORMAL', 'PNEUMONIA'}
    COLORS  = {'NORMAL': '#4CAF50', 'PNEUMONIA': '#F44336'}

    def __init__(self, base_path='datasets', target_size=(224, 224),
                 graphs_dir='graphs', random_state=42,
                 chestxray8_normal_cap=5000,
                 n_splits=5, test_size=0.15):
        """
        Parameters
        ----------
        base_path             : root directory of the datasets
        target_size           : target image dimensions after resize (H, W)
        graphs_dir            : output directory for plots
        random_state          : global seed for reproducibility
        chestxray8_normal_cap : cap on NORMAL samples from ChestX-ray8
        n_splits              : number of StratifiedKFold folds
        test_size             : fraction of the dataset reserved for the holdout test set
        """
        self.base_path             = base_path
        self.target_size           = target_size
        self.graphs_dir            = graphs_dir
        self.random_state          = random_state
        self.chestxray8_normal_cap = chestxray8_normal_cap
        self.n_splits              = n_splits
        self.test_size             = test_size

        self.df               = None
        self.folds            = []
        self.X_test           = None
        self.y_test           = None
        self.train_datagen    = None
        self.val_test_datagen = None
        self._sample_path     = None

        os.makedirs(self.graphs_dir, exist_ok=True)

    def run(self):
        """Runs the full pipeline in order."""
        self._load_metadata()
        self._overview()
        self._plot_class_distribution(suffix='_original')
        self._plot_dimension_distribution()
        self._plot_visual_samples()
        self._remove_corrupted()
        self._stratified_sample()
        self._plot_class_distribution(suffix='_balanced')
        self._plot_preprocessing()
        self._configure_augmentation()
        self._plot_augmentation()
        self._split_kfold()
        self._save_csv()

    # Loading

    def _normalize_class(self, dir_name):
        """
        Unifies variations like 'NORMAL (1)' or 'PNEUMONIA_BACTERIAL'
        into a single label. Returns None for intermediate directories.
        """
        u = dir_name.upper()
        if 'PNEUMONIA' in u:
            return 'PNEUMONIA'
        if 'NORMAL' in u:
            return 'NORMAL'
        return None

    def _load_metadata(self):
        """
        Recursively traverses datasets using os.walk, locating
        NORMAL/PNEUMONIA folders at any level — compatible with
        flat structures (chest_xray) and nested ones with multiple
        intermediate subdirectories (ChestX-ray8 with timestamps).
        """
        print("Loading metadata...")
        records = []

        for dataset in os.listdir(self.base_path):
            dataset_path = os.path.join(self.base_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            for dirpath, _, filenames in os.walk(dataset_path):
                label = self._normalize_class(os.path.basename(dirpath))
                if label is None:
                    continue

                for img_name in filenames:
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    path = os.path.join(dirpath, img_name)
                    img  = cv2.imread(path)

                    if img is None:
                        height, width, channels = None, None, None
                    else:
                        height, width = img.shape[:2]
                        channels      = img.shape[2] if len(img.shape) == 3 else 1

                    records.append({
                        'dataset': dataset, 'label': label,
                        'image': img_name, 'path': path,
                        'height': height, 'width': width, 'channels': channels
                    })

        self.df = pd.DataFrame(records)
        print(f"Total loaded: {len(self.df)} images")
        print(self.df.groupby(['dataset', 'label']).size()
                     .reset_index(name='count').to_string(index=False))

    # EDA

    def _overview(self):
        total = self.df.groupby('dataset').size().reset_index(name='total')
        dist  = self.df['label'].value_counts().reset_index()
        dist.columns = ['label', 'count']

        print("\nTotal per dataset:")
        print(total.to_string(index=False))
        print("\nTotal per class (unified set):")
        print(dist.to_string(index=False))
        print(f"\nTotal images   : {len(self.df)}")
        print(f"Corrupted images: {self.df['height'].isna().sum()}")

    def _plot_class_distribution(self, suffix=''):
        """
        Generates a bar chart per dataset plus one for the unified set.
        The suffix differentiates versions before/after sampling.
        """
        print(f"\nPlotting class distribution{suffix}...")
        datasets = self.df['dataset'].unique()
        n        = len(datasets)

        fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 5))

        for ax, ds in zip(axes[:n], datasets):
            counts = self.df[self.df['dataset'] == ds]['label'].value_counts()
            self._barplot(ax, counts, ds)

        self._barplot(axes[n], self.df['label'].value_counts(), 'UNIFIED SET')

        title = 'Class Distribution per Dataset + Unified Set'
        if suffix:
            title += f' ({suffix.strip("_").capitalize()})'

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/class_distribution{suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _barplot(self, ax, counts, title):
        """Helper to draw a bar plot with absolute counts and percentages."""
        bars  = ax.bar(counts.index, counts.values,
                       color=[self.COLORS[c] for c in counts.index],
                       edgecolor='black', linewidth=0.7)
        total = counts.sum()
        pct   = (counts / total * 100).round(1)

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha='center', va='bottom', fontsize=10)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f'{c}\n({pct[c]}%)' for c in counts.index])

    def _plot_dimension_distribution(self):
        """Histograms of height and width per dataset."""
        print("\nPlotting dimension distribution...")
        datasets  = self.df['dataset'].unique()
        fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 4 * len(datasets)))

        for i, ds in enumerate(datasets):
            df_t  = self.df[self.df['dataset'] == ds].dropna(subset=['height', 'width'])
            ax_h  = axes[i][0] if len(datasets) > 1 else axes[0]
            ax_w  = axes[i][1] if len(datasets) > 1 else axes[1]

            for ax, col, color, label in [
                (ax_h, 'height', 'steelblue',  'Height'),
                (ax_w, 'width',  'darkorange', 'Width'),
            ]:
                ax.hist(df_t[col], bins=30, color=color, edgecolor='black', alpha=0.8)
                ax.set_title(f'[{ds}] {label}')
                ax.set_xlabel('Pixels')
                ax.set_ylabel('Frequency')
                ax.axvline(df_t[col].mean(), color='red', linestyle='--',
                           label=f"Mean: {df_t[col].mean():.0f}px")
                ax.legend()

            print(f"\n  {ds}")
            for col in ['height', 'width']:
                print(f"   {col.capitalize():7} — "
                      f"min: {df_t[col].min():.0f} | max: {df_t[col].max():.0f} | "
                      f"mean: {df_t[col].mean():.1f} | median: {df_t[col].median():.0f}")

        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/dimension_distribution.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_visual_samples(self):
        """Displays a grid of random samples per dataset and class."""
        print("\nGenerating visual samples...")
        for ds in self.df['dataset'].unique():
            classes        = self.df[self.df['dataset'] == ds]['label'].unique()
            n_cols, n_rows = 4, len(classes)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
            if n_rows == 1:
                axes = [axes]

            for row_idx, label in enumerate(classes):
                samples = self.df[
                    (self.df['dataset'] == ds) &
                    (self.df['label']   == label) &
                    (self.df['height'].notna())
                ].sample(min(n_cols, 4), random_state=self.random_state)

                for col_idx, (_, s) in enumerate(samples.iterrows()):
                    img = cv2.cvtColor(cv2.imread(s['path']), cv2.COLOR_BGR2RGB)
                    axes[row_idx][col_idx].imshow(img)
                    axes[row_idx][col_idx].set_title(label, fontsize=10)
                    axes[row_idx][col_idx].axis('off')

                for col_idx in range(len(samples), n_cols):
                    axes[row_idx][col_idx].axis('off')

            plt.suptitle(f'Samples — {ds}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.graphs_dir}/samples_{ds.replace(" ", "_")}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    # Cleaning

    def _remove_corrupted(self):
        """Removes images that OpenCV could not open (height == None)."""
        corrupted = self.df[self.df['height'].isna()]
        print(f"\nCorrupted images: {len(corrupted)}")
        if len(corrupted) > 0:
            print(corrupted[['dataset', 'label', 'image']].to_string(index=False))
            self.df = self.df[self.df['height'].notna()].reset_index(drop=True)
            print(f"Removed. Remaining: {len(self.df)} images.")
        else:
            print("No corrupted images found.")

    # Stratified Sampling

    def _stratified_sample(self):
        """
        Samples only the NORMAL class from ChestX-ray8, which concentrates
        ~60k images and would skew the unified dataset.

        All other datasets are kept intact, preserving their original class
        distributions. Residual imbalance is handled via class_weight during training.
        """
        target_dataset = 'ChestX-ray8'
        target_class   = 'NORMAL'

        mask       = (self.df['dataset'] == target_dataset) & (self.df['label'] == target_class)
        n_original = mask.sum()

        if n_original > self.chestxray8_normal_cap:
            sampled_idx = (
                self.df[mask]
                .sample(self.chestxray8_normal_cap, random_state=self.random_state)
                .index
            )
            self.df = pd.concat([
                self.df[~mask],
                self.df.loc[sampled_idx]
            ]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

            print(f"\nSampling: [{target_dataset}] {target_class} "
                  f"{n_original} → {self.chestxray8_normal_cap} samples")
        else:
            print(f"\nSampling: [{target_dataset}] {target_class} "
                  f"below cap ({n_original}), kept as-is.")

        print(f"Final total: {len(self.df)} images")
        print(self.df.groupby(['dataset', 'label']).size()
                     .reset_index(name='count').to_string(index=False))

    # Preprocessing

    def preprocess_image(self, path, normalize=True):
        """
        Reads, converts to RGB, resizes, and normalizes an image.
        Returns a float32 np.ndarray with values in [0, 1].
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image not found: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                         interpolation=cv2.INTER_AREA)

        if normalize:
            img = img.astype(np.float32) / 255.0

        return img

    def _plot_preprocessing(self):
        """Displays an image before and after preprocessing."""
        print("\nDemonstrating preprocessing...")
        self._sample_path = self.df['path'].iloc[0]
        img_proc          = self.preprocess_image(self._sample_path)
        img_orig          = cv2.cvtColor(cv2.imread(self._sample_path), cv2.COLOR_BGR2RGB)

        print(f"Shape  : {img_proc.shape}")
        print(f"Min/Max: {img_proc.min():.4f} / {img_proc.max():.4f}")
        print(f"Dtype  : {img_proc.dtype}")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_orig)
        plt.title(f'Original\n{img_orig.shape[1]}×{img_orig.shape[0]}px')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_proc)
        plt.title(f'Preprocessed\n{self.target_size[1]}×{self.target_size[0]}px | [0,1]')
        plt.axis('off')
        plt.suptitle('Before × After Preprocessing', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/preprocessing_example.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Augmentation

    def _configure_augmentation(self):
        """
        Defines two generators:
        - train_datagen    : random transformations for training diversity.
        - val_test_datagen : normalization only, for deterministic evaluation.
        """
        print("\nConfiguring Data Augmentation...")

        self.train_datagen = ImageDataGenerator(
            rescale            = 1./255,
            rotation_range     = 15,
            width_shift_range  = 0.05,
            height_shift_range = 0.05,
            zoom_range         = 0.1,
            horizontal_flip    = True,
            brightness_range   = [0.85, 1.15],
            fill_mode          = 'nearest'
        )

        self.val_test_datagen = ImageDataGenerator(rescale=1./255)

    def _plot_augmentation(self):
        """
        Visualizes augmentation examples.
        The generator receives uint8 [0, 255] and applies rescale internally.
        """
        img_uint8       = cv2.cvtColor(cv2.imread(self._sample_path), cv2.COLOR_BGR2RGB)
        sample_expanded = np.expand_dims(img_uint8, axis=0).astype(np.uint8)

        aug_gen = ImageDataGenerator(
            rescale=1./255, rotation_range=15,
            width_shift_range=0.05, height_shift_range=0.05,
            zoom_range=0.1, horizontal_flip=True,
            brightness_range=[0.85, 1.15], fill_mode='nearest'
        )

        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes[0][0].imshow(img_uint8)
        axes[0][0].set_title('Original', fontweight='bold')
        axes[0][0].axis('off')

        aug_iter = aug_gen.flow(sample_expanded, batch_size=1)
        for idx in range(1, 10):
            aug_img  = next(aug_iter)[0]
            row, col = divmod(idx, 5)
            axes[row][col].imshow(np.clip(aug_img, 0, 1))
            axes[row][col].set_title(f'Aug {idx}')
            axes[row][col].axis('off')

        plt.suptitle('Data Augmentation Examples', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.graphs_dir}/augmentation_examples.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    # StratifiedKFold Split

    def _split_kfold(self):
        """
        Two-step split:

        Step 1 — Holdout test set (stratified train_test_split):
            `test_size` % of the total dataset is reserved as a fixed test set.
            Separated BEFORE KFold — never participates in any fold,
            ensuring the final evaluation is unbiased and independent.

        Step 2 — StratifiedKFold on the remaining data:
            The remaining (1 - test_size) is split into `n_splits` folds.
            shuffle=True + random_state fixes the permutation for reproducibility.

            Per fold k:
              - train      : (n_splits - 1) parts  ≈ (1 - test_size) * (K-1)/K
              - validation : 1 part                ≈ (1 - test_size) *    1/K

            Example with n_splits=5, test_size=0.15:
              - test       : 15%
              - train      : 85% * 4/5 = 68%
              - validation : 85% * 1/5 = 17%
        """
        print(f"\nSplitting with StratifiedKFold "
              f"(n_splits={self.n_splits}, test_size={self.test_size}, "
              f"random_state={self.random_state})...")

        X = self.df['path'].values
        y = self.df['label'].values

        # Step 1: holdout test set
        X_dev, self.X_test, y_dev, self.y_test = train_test_split(
            X, y,
            test_size    = self.test_size,
            stratify     = y,
            random_state = self.random_state
        )

        total = len(X)
        print(f"\n  Holdout test : {len(self.X_test)} samples "
              f"({len(self.X_test)/total*100:.1f}%)  "
              f"→ {Counter(self.y_test)}")

        # Step 2: StratifiedKFold on the development set
        skf = StratifiedKFold(
            n_splits     = self.n_splits,
            shuffle      = True,
            random_state = self.random_state
        )

        self.folds = []
        print(f"\n  Folds on development set ({len(X_dev)} samples):\n")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
            fold = {
                'fold'   : fold_idx,
                'X_train': X_dev[train_idx],
                'y_train': y_dev[train_idx],
                'X_val'  : X_dev[val_idx],
                'y_val'  : y_dev[val_idx],
            }
            self.folds.append(fold)

            c_train = Counter(y_dev[train_idx])
            c_val   = Counter(y_dev[val_idx])
            print(f"  Fold {fold_idx}:")
            print(f"    train      : {len(train_idx):>6} samples  → {dict(c_train)}")
            print(f"    validation : {len(val_idx):>6} samples  → {dict(c_val)}")

    # Persistence

    def _save_csv(self):
        """
        Saves folds and test set to CSV for reuse in the training step
        without re-running the pipeline.

        Columns: path | label | split | fold
        fold = -1 for test set samples.
        """
        parts = [pd.DataFrame({
            'path' : self.X_test,
            'label': self.y_test,
            'split': 'test',
            'fold' : -1
        })]

        for f in self.folds:
            parts.append(pd.DataFrame({
                'path' : f['X_train'],
                'label': f['y_train'],
                'split': 'train',
                'fold' : f['fold']
            }))
            parts.append(pd.DataFrame({
                'path' : f['X_val'],
                'label': f['y_val'],
                'split': 'val',
                'fold' : f['fold']
            }))

        df_splits = pd.concat(parts, ignore_index=True)

        save_path = os.path.join(self.base_path, '..', 'splits_dataset.csv')
        df_splits.to_csv(save_path, index=False)
        print(f"\nSplits saved to: {os.path.abspath(save_path)}")

        summary = (df_splits[df_splits['fold'] != -1]
                   .groupby(['fold', 'split'])
                   .size()
                   .reset_index(name='n'))
        print("\n  Summary per fold:")
        print(summary.to_string(index=False))
        print(f"\n  Test set: {len(self.X_test)} samples (fold = -1)")