"""``keras.ops`` port of ``hepattn.utils.scaling`` (that module imports torch).

Same YAML format and the same arithmetic as the torch version. One deliberate difference:
``FeatureScaler.transform`` / ``inverse_transform`` return a new dict instead of writing into the
argument, because JAX arrays are immutable and traced code must not rely on mutation.
"""

from collections import defaultdict
from pathlib import Path

import keras
import yaml


class VarTransform:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.type = self.config.get("type", "std")
        self.fn = self.config.get("fn", None)

        if self.type == "std":
            self.shift = self.config.get("mean", 0.0)
            self.scale = self.config.get("std", 1.0)
        elif self.type == "min_max":
            self.shift = self.config.get("min", 0.0)
            self.scale = self.config.get("max", 1.0) - self.shift
        elif self.type == "min_max_sym":
            min_ = self.config.get("min", 0.0)
            max_ = self.config.get("max", 1.0)
            self.shift = (max_ + min_) / 2
            self.scale = (max_ - min_) / 2
        if self.type not in {"std", "min_max", "min_max_sym"}:
            raise ValueError(f"Unknown scaling type {self.type}")
        if self.fn not in {None, "log", "log1p", "sqrt"}:
            raise ValueError(f"Unknown scaling fn {self.fn}")

    def transform(self, x, shift=None, scale=None):
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        if self.fn == "log":
            x = keras.ops.log(x)
        elif self.fn == "log1p":
            x = keras.ops.log1p(x)
        elif self.fn == "sqrt":
            x = keras.ops.sqrt(x)

        return (x - shift) / scale

    def inverse_transform(self, x, shift=None, scale=None):
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = x * scale + shift
        if self.fn == "log":
            return keras.ops.exp(x)
        if self.fn == "log1p":
            return keras.ops.expm1(x)
        if self.fn == "sqrt":
            return keras.ops.square(x)

        return x


def get_empty_transform() -> VarTransform:
    """Identity transform (std scaling with mean 0, std 1, no fn)."""
    return VarTransform("", {"type": "std", "mean": 0.0, "std": 1.0})


class FeatureScaler:
    def __init__(self, scale_dict_path: str | Path):
        """Read per-feature scaling parameters from a YAML file (same format as the torch version)."""
        with Path(scale_dict_path).open() as f:
            self.scale_dict = yaml.safe_load(f)
        self.transforms = defaultdict(get_empty_transform)
        for name, config in self.scale_dict.items():
            self.transforms[name] = VarTransform(name, config)

    def __getitem__(self, name: str) -> VarTransform:
        return self.transforms[name]

    def transform(self, x: dict) -> dict:
        """Return a new dict with every known feature transformed; other keys pass through."""
        return {k: self.transforms[k].transform(v) if k in self.transforms else v for k, v in x.items()}

    def inverse_transform(self, x: dict) -> dict:
        """Return a new dict with every known feature inverse-transformed; other keys pass through."""
        return {k: self.transforms[k].inverse_transform(v) if k in self.transforms else v for k, v in x.items()}
