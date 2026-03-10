# `tutorial_medmnist3d_cnn_tensorflow.py` walkthrough

This tutorial trains a small **3D CNN** in Keras (TensorFlow backend) on a MedMNIST 3D dataset.

## What dataset is used?

By default, the script uses:

- `dataset_flag = "nodulemnist3d"`
- MedMNIST class: `NoduleMNIST3D`

This is a dataset of 3D thoracic CT nodule crops (from LIDC-IDRI) packaged by MedMNIST.

## What task is being solved?

For `nodulemnist3d`, the MedMNIST task is:

- `task = "binary-class"`
- labels:
  - `0 -> benign`
  - `1 -> malignant`

So this tutorial is a **binary classification** problem: predict whether each nodule volume is benign or malignant.

## Data characteristics (default setting)

- Input volume size: `28 x 28 x 28` (`size=28`)
- Channels: 1 (grayscale CT intensity)
- Splits (from MedMNIST metadata):
  - train: 1158
  - val: 165
  - test: 310

The script reshapes data to Keras Conv3D format:

- from `(N, D, H, W)` to `(N, D, H, W, C)` with `C=1`

## Script flow

1. Read configuration from `ExperimentConfig`.
2. Load MedMNIST metadata via `INFO[dataset_flag]`.
3. Create `data_root` directory if missing.
4. Download/load `train`, `val`, `test` splits.
5. Normalize voxel values from `[0,255]` to `[0,1]`.
6. Build a compact 3D CNN:
   - Conv3D(16) -> Conv3D(32) -> Conv3D(64)
   - BatchNorm / MaxPool
   - GlobalAveragePooling3D + Dense head
7. Train with AdamW and early stopping.
8. Evaluate on test set:
   - loss
   - accuracy
   - AUC (binary positive-class AUC)
9. Print sample predictions with class names and confidence.

## How to run

From repo root:

```bash
python tutorial/tutorial_medmnist3d_cnn_tensorflow.py
```

## Useful config fields

Edit `CONFIG = ExperimentConfig()` values directly in the script:

- `dataset_flag`: choose another MedMNIST 3D dataset (e.g. `adrenalmnist3d`, `fracturemnist3d`)
- `size`: `28` or `64`
- `epochs`, `batch_size`, `learning_rate`, `weight_decay`
- `data_root`: where `.npz` files are stored
- `save_model_path`: optional checkpoint path

## Notes

- The tutorial enforces TensorFlow backend (`KERAS_BACKEND=tensorflow`).
- If you switch to a non-binary dataset, AUC is computed in one-vs-rest mode.
