# ruff: noqa: E501
"""R3 fix round 2 mutants: id -> (root, [(file, old, new), ...]); root 'wt' = the rt/r3 tree, 'merged' = rt/r3 + rt/r2."""

N = "src/hepattn/keras_native/"
T = "tests/keras_native/"
HOST_SORT = (
    'x["key_sort_idx"] = sort_index(sort_value, x["key_valid"])',
    'import numpy as _np; x["key_sort_idx"] = keras.ops.convert_to_tensor(_np.argsort(_np.where(_np.asarray(x["key_valid"]), _np.asarray(sort_value), _np.inf), axis=-1, kind="stable").astype("int32"))',
)
MUTANTS = {
    # SwiGLU leaf rule = the hybrid twin (R4a)
    "g02_packed_glorot": (
        "wt",
        [
            (
                N + "dense.py",
                '            if init == "keras":\n                return {}\n',
                '            if init == "keras":\n                if self.gate and fan_in in fan_ins[:-1]:\n                    lim = math.sqrt(6.0 / (fan_in + 2 * self.hidden_sizes[fan_ins.index(fan_in)]))\n                    return {"kernel_initializer": keras.initializers.RandomUniform(-lim, lim)}\n                return {}\n',
            )
        ],
    ),
    "g03_gate_name": ("wt", [(N + "dense.py", 'name=sub(f"hidden{i}_x2")', 'name=sub(f"hidden{i}_gate")')]),
    "g03b_x1_unsuffixed": ("wt", [(N + "dense.py", 'x1_suffix = "_x1" if self.gate else ""', 'x1_suffix = ""')]),
    "g04_ellipsis": ("wt", [(N + "dense.py", 'GLU_EQUATION = "bnc,bnc->bnc"', 'GLU_EQUATION = "...c,...c->...c"')]),
    "g05_no_fold": ("wt", [(N + "dense.py", "        if len(x1.shape) == 3:\n", "        if True:\n")]),
    # model-level jit row (reviewer's y11)
    "y11_host_sort_model": ("wt", [(N + "maskformer.py", *HOST_SORT)]),
    # sorting reference: sort_index itself wrong (descending) -> a self-referential check cannot see it
    "g06_sort_descending": (
        "wt",
        [(N + "encoder.py", "    return keras.ops.argsort(value, axis=-1)\n", "    return keras.ops.argsort(-value, axis=-1)\n")],
    ),
    # q_sa core (merged tree)
    "g01_qsa_masked_core": ("merged", [(N + "decoder.py", '"q_sa": "project_unmasked"', '"q_sa": "project"')]),
    # projection shape gate (merged tree): kv_ca tail dropped; linformer_k ignored (k=256)
    "g07_no_tail": ("merged", [(N + "masked_linformer.py", "self.has_tail = renorm and seq_len > n", "self.has_tail = False")]),
    # the same mutants against the f4f37db tests (before this round)
    "o06_sort_descending_old_test": (
        "old",
        MUTANTS_G06 := [(N + "encoder.py", "    return keras.ops.argsort(value, axis=-1)\n", "    return keras.ops.argsort(-value, axis=-1)\n")],
    ),
    "o11_host_sort_old_tests": ("old", [(N + "maskformer.py", *HOST_SORT)]),
    # encoder core pinned to project with the mask; the unmasked core drops the renorm (shapes unchanged: 168 = 168)
    "g08_enc_unmasked": (
        "merged",
        [(N + "encoder.py", 'attn_kwargs["linformer_core"] = "project"', 'attn_kwargs["linformer_core"] = "project_unmasked"')],
    ),
}
