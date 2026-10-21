r"""S7: hls4ml C-simulation bit-exactness of the native export models, on Linux (macOS cannot build hls4ml's headers).

Two steps, two machines:

    # 1. macOS (or any), .venv-jax with hls4ml: build every model of tests/keras_native/test_export.py that has a csim
    #    gate (toy MLP, toy attention core, one model per HGQ2 leaf kind of the org QAT model with the kept / silu export
    #    forms, the classification head in its LUT form) and save each as .keras with its inputs
    env -u KERAS_BACKEND PYTHONPATH=src .venv-jax/bin/python docs/hgq/port_log/S7_csim.py export <dir>

    # 2. Linux (falcon): keras + hgq2 + jax (CPU) + hls4ml + da4ml only; PYTHONPATH=src for the table functions
    #    (masked_linformer.rsqrt_floor) the .keras files name
    KERAS_BACKEND=jax PYTHONPATH=src python docs/hgq/port_log/S7_csim.py csim <dir>

Inputs are those of the model's domain, as in the pytest csim test (``test_export.domain_inputs``): 0/1 masks for the
kept einsums; for the renorm rsqrt table the kept energy fraction in [0, 1), with the floor point 0 in the first column
of every row. No training call before the export: the models hold the converted checkpoint's state.

``csim`` loads each model, runs keras and hls4ml's C simulation (Vitis, io_parallel, ``compile()``) on the stored
inputs and prints max |hls - keras| and the number of differing elements; it exits 1 unless every row is exactly equal,
except the ``CONTROL`` rows (the inline sigmoid as trained), which must differ, and the rsqrt floor points (input 0),
where keras must give 2**10 and hls4ml the lane maximum 2**10 - 2**-14 (one LSB apart; ``test_export.assert_rsqrt_csim``
explains why, ``test_rsqrt_floor_does_not_reach_the_core_outputs`` that it does not reach the core outputs); the rsqrt
row's ``n_diff`` counts the other points. A control perturbs the first model's C-simulation input by 3 least
significant bits of its input quantizer (3 x 2**-8) on one element and requires a difference (so an all-zero
comparison is not vacuous). Results go to ``<dir>/csim.json``.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")  # before keras is imported (the native models run on JAX)


def export(out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tests.keras_native import test_export as T  # noqa: PLC0415

    from hepattn.experiments.clic import convert_to_native as C  # noqa: PLC0415
    from hepattn.keras_native.export import build_functional_dense, leaf_model  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    _, org, _ = C.convert([T.FIX / "config.yaml"], T.FIX / "model.ckpt")
    rng = np.random.default_rng(6)
    models = {"toy_mlp": T.build_toy_mlp(), "toy_attention_core": T.build_toy_attention_core()}
    models |= T.export_models(org)
    head = org.tasks[0].net
    shape = tuple(head.hidden[0]._build_shapes_dict["input_shape"][1:])  # noqa: SLF001
    models["classification_head (lut form)"] = build_functional_dense(head, input_shape=shape, name="clf_head_lut", activations="lut")
    models["CONTROL QDense:sigmoid: as trained"] = leaf_model(org.encoder.layers[1].attn.fn.value_residual_mix)
    manifest = []
    for i, (key, model) in enumerate(models.items()):
        inputs = T.domain_inputs(key, model, rng)
        inputs = inputs if isinstance(inputs, list) else [inputs]
        stem = f"m{i:02d}"
        model.save(str(out / f"{stem}.keras"))
        np.savez(out / f"{stem}.npz", *inputs)
        np.save(out / f"{stem}_keras_out.npy", np.asarray(model(inputs if len(inputs) > 1 else inputs[0], training=False)))
        manifest.append({"stem": stem, "key": key, "inputs": [list(x.shape) for x in inputs]})
        print(stem, key, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def csim(root: Path) -> int:
    import hls4ml  # noqa: PLC0415
    import keras  # noqa: PLC0415

    from hepattn.keras_native.export import load_keras_model  # noqa: PLC0415

    manifest = json.loads((root / "manifest.json").read_text())
    rows, bad = [], 0
    for entry in manifest:
        stem = entry["stem"]
        model = load_keras_model(root / f"{stem}.keras")
        with np.load(root / f"{stem}.npz") as data:
            inputs = [data[f"arr_{i}"] for i in range(len(data.files))]
        x = inputs if len(inputs) > 1 else inputs[0]
        y_keras = np.asarray(model(x, training=False))
        y_export = np.load(root / f"{stem}_keras_out.npy")  # keras at export time (the training machine)
        t0 = time.perf_counter()
        hls = hls4ml.converters.convert_from_keras_model(model, output_dir=str(root / f"hls_{stem}"), backend="Vitis", io_type="io_parallel")
        hls.compile()
        y_hls = np.asarray(hls.predict([np.ascontiguousarray(a) for a in inputs] if len(inputs) > 1 else np.ascontiguousarray(x))).reshape(
            y_keras.shape
        )
        diff = np.abs(y_hls.astype(np.float64) - y_keras)
        floor_ok = True
        if "rsqrt" in entry["key"]:  # the floor points (input 0), characterized in test_export.assert_rsqrt_csim
            floor = x == 0
            floor_ok = bool(floor.any() and np.all(y_keras[floor] == 2.0**10) and np.all(y_hls[floor] == 2.0**10 - 2.0**-14))
            diff = np.where(floor, 0.0, diff)
        row = {
            "stem": stem,
            "key": entry["key"],
            "max_abs": float(diff.max()),
            "n_diff": int((diff > 0).sum()),
            "size": int(diff.size),
            "y_std": float(y_keras.std()),
            "keras_vs_export_max_abs": float(np.abs(y_keras - y_export).max()),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        if "rsqrt" in entry["key"]:
            row |= {"floor_points": int((x == 0).sum()), "floor_as_characterized": floor_ok}
        if stem == manifest[0]["stem"]:  # control: 3 LSB (LSB 2**-8 for the toy datalane f0=8) on one input element
            xc = np.array(x, copy=True)
            xc.flat[0] += 2.0**-8 * (1 if xc.flat[0] < 0 else -1) * 3
            yc = np.asarray(hls.predict(np.ascontiguousarray(xc))).reshape(y_keras.shape)
            row["control_n_diff"] = int((yc != y_hls).sum())
            bad += row["control_n_diff"] == 0
        expect_equal = not entry["key"].startswith("CONTROL")  # a CONTROL model must differ (hls4ml's own table)
        bad += (row["n_diff"] > 0) == expect_equal or not np.isfinite(y_keras).all() or not floor_ok
        rows.append(row)
        print(json.dumps(row), flush=True)
    (root / "csim.json").write_text(json.dumps({"hls4ml": hls4ml.__version__, "keras": keras.__version__, "rows": rows}, indent=1))
    print(f"{len(rows)} models, {bad} not bit-exact (or control failed)")
    return int(bad > 0)


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "export":
        export(path)
    else:
        sys.exit(csim(path))
