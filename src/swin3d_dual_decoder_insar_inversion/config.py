from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputConfig:
    path: str
    variable: str
    format: str = "auto"
    dim_order: tuple[str, str, str] | None = None
    transpose_order: tuple[int, int, int] | None = None
    time_name: str | None = "time"
    lat_name: str | None = "lat"
    lon_name: str | None = "lon"
    mask_variable: str | None = None


@dataclass(frozen=True)
class TilingConfig:
    window_size: int = 12
    tile_size: int = 64
    stride: int = 64
    min_valid_fraction: float = 0.05


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "runs/default"
    seed: int = 42
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4
    val_fraction: float = 0.2
    shuffle_train: bool = False
    lambda_forward: float = 1.0
    lambda_spatial_s0: float = 0.01
    lambda_spatial_sg: float = 0.01
    lambda_temporal_s0: float = 0.005
    lambda_temporal_sg: float = 0.005
    num_workers: int = 0


@dataclass(frozen=True)
class ModelConfig:
    base_dim: int = 32
    time_patch: int = 2
    spatial_patch: int = 4
    num_heads: int = 4
    window_size: tuple[int, int, int] = (3, 4, 4)
    merge_scale: tuple[int, int, int] = (1, 2, 2)


@dataclass(frozen=True)
class PhysicsConfig:
    E: float = 1e9
    nu: float = 0.25
    rho_w: float = 1000.0
    g: float = 9.81
    alpha: float = 0.8
    Hg: float = 150.0
    Seff: float = 0.2
    dx: float = 10000.0
    dy: float = 10000.0
    a_load: float = 3000.0
    a_poro: float = 3000.0


@dataclass(frozen=True)
class InversionConfig:
    input: InputConfig
    tiling: TilingConfig
    training: TrainingConfig
    model: ModelConfig
    physics: PhysicsConfig

    def to_dict(self) -> dict:
        return asdict(self)


def _tuple_or_none(value):
    if value is None:
        return None
    return tuple(value)


def load_config(path: str | Path) -> InversionConfig:
    payload = json.loads(Path(path).read_text())
    return InversionConfig(
        input=InputConfig(
            path=payload["input"]["path"],
            variable=payload["input"]["variable"],
            format=payload["input"].get("format", "auto"),
            dim_order=_tuple_or_none(payload["input"].get("dim_order")),
            transpose_order=_tuple_or_none(payload["input"].get("transpose_order")),
            time_name=payload["input"].get("time_name"),
            lat_name=payload["input"].get("lat_name"),
            lon_name=payload["input"].get("lon_name"),
            mask_variable=payload["input"].get("mask_variable"),
        ),
        tiling=TilingConfig(**payload.get("tiling", {})),
        training=TrainingConfig(**payload.get("training", {})),
        model=ModelConfig(
            base_dim=payload.get("model", {}).get("base_dim", 32),
            time_patch=payload.get("model", {}).get("time_patch", 2),
            spatial_patch=payload.get("model", {}).get("spatial_patch", 4),
            num_heads=payload.get("model", {}).get("num_heads", 4),
            window_size=tuple(payload.get("model", {}).get("window_size", [3, 4, 4])),
            merge_scale=tuple(payload.get("model", {}).get("merge_scale", [1, 2, 2])),
        ),
        physics=PhysicsConfig(**payload.get("physics", {})),
    )

