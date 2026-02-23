"""Thin adapter around the uploaded AuralMind mastering script.

Goals:
- Dynamically import the user-provided mastering script.
- Expose a stable interface to list presets and run mastering.
- Validate and constrain LLM-provided override values before applying them.
- Keep the server code independent from the script's internal implementation details.

This adapter expects the script to expose:
  - get_presets() -> dict[str, Preset]
  - master(target_path, out_path, preset, reference_path=None, report_path=None, *, out_subtype=None, dither=None, dither_seed=0) -> dict

The uploaded v7.3 expert script matches that signature.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import importlib.util
import inspect
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class EngineNotReadyError(RuntimeError):
    pass


class AuralMindAdapter:
    """Runtime-safe wrapper for dynamic loading + preset overrides."""

    #: Fields the LLM can tweak safely. You can expand this list over time.
    SAFE_OVERRIDE_SPECS: dict[str, dict[str, Any]] = {
        "target_lufs": {"type": float, "min": -16.0, "max": -7.0},
        "ceiling_dbfs": {"type": float, "min": -2.0, "max": -0.1},
        "match_strength": {"type": float, "min": 0.0, "max": 1.0},
        "max_eq_db": {"type": float, "min": 0.5, "max": 12.0},
        "eq_smooth_hz": {"type": float, "min": 40.0, "max": 400.0},
        "deess_threshold_db": {"type": float, "min": -40.0, "max": 0.0},
        "deess_mix": {"type": float, "min": 0.0, "max": 1.0},
        "glow_drive_db": {"type": float, "min": 0.0, "max": 6.0},
        "glow_mix": {"type": float, "min": 0.0, "max": 1.0},
        "width_mid": {"type": float, "min": 0.8, "max": 1.4},
        "width_hi": {"type": float, "min": 0.8, "max": 1.5},
        "microshift_ms": {"type": float, "min": 0.0, "max": 2.0},
        "microshift_mix": {"type": float, "min": 0.0, "max": 0.5},
        "enable_softclip": {"type": bool},
        "softclip_drive_db": {"type": float, "min": 0.0, "max": 8.0},
        "softclip_mix": {"type": float, "min": 0.0, "max": 1.0},
        "enable_microdetail": {"type": bool},
        "microdetail_amount": {"type": float, "min": 0.0, "max": 1.0},
        "microdetail_max_boost_db": {"type": float, "min": 0.0, "max": 8.0},
        "microdetail_mix": {"type": float, "min": 0.0, "max": 1.0},
        "enable_movement": {"type": bool},
        "movement_amount": {"type": float, "min": 0.0, "max": 0.5},
        "enable_hooklift": {"type": bool},
        "hooklift_auto": {"type": bool},
        "hooklift_mix": {"type": float, "min": 0.0, "max": 1.0},
        "enable_transient_sculpt": {"type": bool},
        "transient_sculpt_boost_db": {"type": float, "min": 0.0, "max": 6.0},
        "transient_sculpt_mix": {"type": float, "min": 0.0, "max": 1.0},
        "transient_sculpt_crest_guard_db": {"type": float, "min": 8.0, "max": 30.0},
        "transient_sculpt_decay_ms": {"type": float, "min": 1.0, "max": 50.0},
        "warmth": {"type": float, "min": -1.0, "max": 1.0},
        "limiter_oversample": {"type": int, "min": 1, "max": 16},
        "limiter_attack_ms": {"type": float, "min": 0.1, "max": 20.0},
        "limiter_release_ms": {"type": float, "min": 5.0, "max": 500.0},
        "limiter_stereo_link": {"type": float, "min": 0.0, "max": 1.0},
        "governor_gr_limit_db": {"type": float, "min": -8.0, "max": -0.1},
        "governor_lufs_tolerance": {"type": float, "min": 0.1, "max": 2.0},
        "governor_comp_db": {"type": float, "min": 0.0, "max": 10.0},
        "enable_stem_separation": {"type": bool},
        "demucs_device": {"type": str, "choices": ["cpu", "cuda", "mps"]},
        "out_subtype": {"virtual": True, "type": str, "choices": ["PCM_16", "PCM_24", "FLOAT"]},
        "dither": {"virtual": True, "type": bool},
    }

    def __init__(self, script_path: str):
        self.script_path = str(script_path)
        self._mod = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._mod is not None:
                return self._mod

            script = Path(self.script_path).resolve()
            if not script.exists():
                raise EngineNotReadyError(f"AuralMind script not found: {script}")

            mod_name = f"auralmind_user_script_{abs(hash(str(script)))}"
            spec = importlib.util.spec_from_file_location(mod_name, script)
            if spec is None or spec.loader is None:
                raise EngineNotReadyError(f"Could not load module spec from {script}")

            module = importlib.util.module_from_spec(spec)
            # Important for dataclasses + __module__ resolution in the uploaded script.
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]

            missing = [n for n in ("get_presets", "master") if not hasattr(module, n)]
            if missing:
                raise EngineNotReadyError(
                    f"Loaded script is missing required symbols: {', '.join(missing)}"
                )

            self._mod = module
            return self._mod

    def reload(self) -> dict[str, Any]:
        with self._lock:
            self._mod = None
        mod = self._load()
        return {
            "ok": True,
            "module": getattr(mod, "__name__", "unknown"),
            "script_path": self.script_path,
            "master_signature": str(inspect.signature(mod.master)),
            "preset_count": len(mod.get_presets()),
        }

    @property
    def module(self):
        return self._load()

    def get_presets(self) -> dict[str, Any]:
        mod = self.module
        presets = mod.get_presets()
        out: dict[str, Any] = {}
        for name, preset in presets.items():
            if is_dataclass(preset):
                try:
                    out[name] = asdict(preset)
                except Exception:
                    # Fallback if any nested object isn't trivially serializable
                    out[name] = {k: getattr(preset, k) for k in getattr(preset, "__dataclass_fields__", {})}
            else:
                out[name] = {"repr": repr(preset)}
        return out

    def get_preset_names(self) -> list[str]:
        return sorted(self.get_presets().keys())

    def get_preset(self, name: str):
        mod = self.module
        presets = mod.get_presets()
        if name not in presets:
            raise KeyError(f"Unknown preset '{name}'. Available: {', '.join(sorted(presets))}")
        return presets[name]

    def get_preset_details(self, name: str) -> dict[str, Any]:
        presets = self.get_presets()
        if name not in presets:
            raise KeyError(f"Unknown preset '{name}'")
        return presets[name]

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "y", "on"}:
                return True
            if v in {"false", "0", "no", "n", "off"}:
                return False
        raise ValueError(f"Cannot coerce {value!r} to bool")

    def validate_overrides(self, overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Validate LLM/user overrides against a safe allowlist.

        Returns only validated engine-field overrides (virtual args like out_subtype/dither
        are handled separately by `split_virtual_master_args`).
        """
        if not overrides:
            return {}

        validated: dict[str, Any] = {}
        for key, raw in overrides.items():
            spec = self.SAFE_OVERRIDE_SPECS.get(key)
            if spec is None:
                raise ValueError(
                    f"Override '{key}' is not allowed. Allowed keys: {', '.join(sorted(self.SAFE_OVERRIDE_SPECS))}"
                )
            if spec.get("virtual"):
                # Virtual keys are not dataclass fields; handled in split_virtual_master_args().
                continue

            t = spec.get("type")
            if t is bool:
                value = self._coerce_bool(raw)
            elif t is int:
                value = int(raw)
            elif t is float:
                value = float(raw)
            elif t is str:
                value = str(raw)
            else:
                value = raw

            if "choices" in spec and value not in spec["choices"]:
                raise ValueError(f"{key} must be one of {spec['choices']}, got {value!r}")
            if "min" in spec and value < spec["min"]:
                raise ValueError(f"{key} must be >= {spec['min']}, got {value}")
            if "max" in spec and value > spec["max"]:
                raise ValueError(f"{key} must be <= {spec['max']}, got {value}")

            validated[key] = value
        return validated

    def split_virtual_master_args(self, overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
        if not overrides:
            return {}
        out: dict[str, Any] = {}
        for key in ("out_subtype", "dither"):
            if key not in overrides:
                continue
            spec = self.SAFE_OVERRIDE_SPECS[key]
            raw = overrides[key]
            if spec["type"] is bool:
                value = self._coerce_bool(raw)
            else:
                value = str(raw)
            if "choices" in spec and value not in spec["choices"]:
                raise ValueError(f"{key} must be one of {spec['choices']}, got {value!r}")
            out[key] = value
        return out

    def apply_overrides(self, preset_obj: Any, overrides: Optional[dict[str, Any]]):
        if not overrides:
            return preset_obj
        valid = self.validate_overrides(overrides)
        if not valid:
            return preset_obj

        # Only apply fields that exist on the dataclass object.
        present = set(getattr(preset_obj, "__dataclass_fields__", {}).keys())
        apply_dict = {k: v for k, v in valid.items() if k in present}
        if not apply_dict:
            return preset_obj
        return replace(preset_obj, **apply_dict)

    def run_master(
        self,
        *,
        target_path: str,
        out_path: str,
        preset_name: str,
        reference_path: Optional[str] = None,
        report_path: Optional[str] = None,
        overrides: Optional[dict[str, Any]] = None,
        dither_seed: int = 0,
    ) -> dict[str, Any]:
        mod = self.module
        base_preset = self.get_preset(preset_name)

        # Practical default for Heroku-sized boxes / CPU-only:
        # If user doesn't explicitly request stems, turn them off (demucs is optional + heavy).
        normalized_overrides = dict(overrides or {})
        normalized_overrides.setdefault("enable_stem_separation", False)

        preset = self.apply_overrides(base_preset, normalized_overrides)
        virtual_args = self.split_virtual_master_args(normalized_overrides)

        result = mod.master(
            target_path=str(target_path),
            out_path=str(out_path),
            preset=preset,
            reference_path=str(reference_path) if reference_path else None,
            report_path=str(report_path) if report_path else None,
            out_subtype=virtual_args.get("out_subtype"),
            dither=virtual_args.get("dither"),
            dither_seed=int(dither_seed),
        )

        # Ensure JSON-serializable response for MCP/API.
        return json.loads(json.dumps(result, default=str))
