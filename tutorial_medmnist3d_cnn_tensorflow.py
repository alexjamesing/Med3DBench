"""IDE-friendly 3D CNN benchmark on MedMNIST radiology volumes (Keras 3 + TF backend).

Default task:
    NoduleMNIST3D (thoracic CT nodules) -> benign vs malignant.

Requirements:
    pip install medmnist tensorflow
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
from keras import layers
from sklearn.metrics import roc_auc_score

try:
    import medmnist
    from medmnist import INFO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "medmnist is required. Install it with: pip install medmnist"
    ) from exc


@dataclass
class ExperimentConfig:
    # Data selection:
    # - `dataset_flag` maps to an entry in `medmnist.INFO`.
    # - Default is NoduleMNIST3D (3D CT nodules, benign vs malignant).
    dataset_flag: str = "nodulemnist3d"
    # Directory where MedMNIST .npz files are stored/downloaded.
    data_root: str = "./data/medmnist"
    # Volumetric resolution. For this dataset, valid values are 28 or 64.
    size: int = 64

    # Optimization
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Reproducibility / outputs
    seed: int = 42
    save_model_path: str | None = None
    preview_samples: int = 8


# Edit these values directly in VS Code before running.
CONFIG = ExperimentConfig()


def load_split_arrays(config: ExperimentConfig, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one MedMNIST3D split and return channels-last arrays for Keras."""
    info = INFO[config.dataset_flag]
    data_class = getattr(medmnist, info["python_class"])

    # Recent MedMNIST versions require `root` to already exist.
    # We proactively create it so `download=True` can populate it.
    data_root = Path(config.data_root).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    # `split` is one of: train / val / test.
    # `download=True` fetches the .npz file only when missing.
    dataset = data_class(
        split=split,
        root=str(data_root),
        download=True,
        size=config.size,
    )

    # Raw pixel intensities are uint8 in [0, 255].
    # Normalize to float32 in [0, 1] for stable optimization.
    x = dataset.imgs.astype("float32") / 255.0
    # Labels are stored with shape (N, 1); flatten to (N,) for sparse CE.
    y = dataset.labels.astype("int64").reshape(-1)

    # Conv3D in Keras uses channels-last by default: (N, D, H, W, C)
    x = np.expand_dims(x, axis=-1)
    return x, y


def load_data(config: ExperimentConfig) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], dict]:
    info = INFO[config.dataset_flag]
    x_train, y_train = load_split_arrays(config, split="train")
    x_val, y_val = load_split_arrays(config, split="val")
    x_test, y_test = load_split_arrays(config, split="test")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), info


def build_3d_cnn(input_shape: tuple[int, int, int, int], num_classes: int) -> keras.Model:
    # Simple baseline architecture:
    # 3 Conv3D blocks -> global pooling -> small MLP classification head.
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv3D(16, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="small_3d_cnn")


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # AUC requires at least two classes in ground truth.
    # This guard avoids runtime errors on tiny/filtered evaluation sets.
    num_classes = y_prob.shape[1]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    # Binary: use score for positive class.
    if num_classes == 2:
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    # Multi-class: one-vs-rest AUC over one-hot labels.
    y_onehot = np.eye(num_classes, dtype=np.float32)[y_true]
    return float(roc_auc_score(y_onehot, y_prob, multi_class="ovr"))


def preview_predictions(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_map: dict[str, str],
    n_samples: int,
) -> None:
    n = min(n_samples, len(x_test))
    prob = model.predict(x_test[:n], verbose=0)
    pred = prob.argmax(axis=1)

    print("Sample predictions:")
    for idx in range(n):
        true_y = int(y_test[idx])
        pred_y = int(pred[idx])
        confidence = float(prob[idx, pred_y])
        print(
            f"  sample={idx:02d} true={label_map[str(true_y)]} "
            f"pred={label_map[str(pred_y)]} confidence={confidence:.3f}"
        )


def main(config: ExperimentConfig = CONFIG) -> None:
    # The tutorial is intentionally pinned to TensorFlow backend behavior.
    if keras.backend.backend() != "tensorflow":
        raise RuntimeError(
            f"Expected TensorFlow backend, got: {keras.backend.backend()!r}"
        )

    # Seed Python/Numpy/TF through Keras utility for reproducible runs.
    keras.utils.set_random_seed(config.seed)

    (x_train, y_train), (x_val, y_val), (x_test, y_test), info = load_data(config)
    num_classes = len(info["label"])

    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Dataset: {config.dataset_flag}")
    print(f"Description: {info['description']}")
    print(f"Task: {info['task']} | Labels: {info['label']}")
    print(f"Shapes: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    model = build_3d_cnn(input_shape=x_train.shape[1:], num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
        # SparseCategoricalCrossentropy expects integer labels (not one-hot).
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks: list[keras.callbacks.Callback] = [
        # Stops training when validation loss plateaus and restores best weights.
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]
    if config.save_model_path:
        out_path = Path(config.save_model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(out_path),
                monitor="val_loss",
                save_best_only=True,
            )
        )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(
        x_test,
        y_test,
        batch_size=config.batch_size,
        verbose=0,
    )
    test_prob = model.predict(x_test, batch_size=config.batch_size, verbose=0)
    test_auc = compute_auc(y_test, test_prob)
    print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")

    preview_predictions(
        model=model,
        x_test=x_test,
        y_test=y_test,
        label_map=info["label"],
        n_samples=config.preview_samples,
    )


if __name__ == "__main__":
    main()
