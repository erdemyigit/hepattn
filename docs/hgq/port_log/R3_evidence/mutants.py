"""R3 mutants: (id, file, old, new). Each old string must occur exactly once in the pristine copy."""

N = "src/hepattn/keras_native/"
MUTANTS = [
    ("m01_fln_affine", N + "norm.py", "    affine = norm not in NON_AFFINE_NORMS\n", "    affine = True\n"),
    (
        "m02_placement_ignored",
        N + "norm.py",
        '    dense_post_norm = is_hybrid_subsequent if dense_norm_placement == "hybridnorm" else not is_hybrid_subsequent\n',
        "    dense_post_norm = is_hybrid_subsequent\n",
    ),
    ("m03_order_ignored", N + "decoder.py", '        if self.query_update_order == "ca_sa_dense":\n', "        if False:\n"),
    (
        "m04_norm_kv_ignored",
        N + "decoder.py",
        "        if self.self_attn_norm_kv:\n            return self.q_sa(q, q_mask=q_mask, training=training)\n",
        "",
    ),
    (
        "m05_sort_idx_everywhere",
        N + "decoder.py",
        "            kv = self.kv_ca(kv, k=q, v=q, attn_mask=mask_t, q_mask=kv_mask, kv_mask=q_mask, training=training)",
        "            kv = self.kv_ca(kv, k=q, v=q, attn_mask=mask_t, q_mask=kv_mask, kv_mask=q_mask, kv_sort_idx=kv_sort_idx, training=training)",
    ),
    ("m06_sort_idx_dropped", N + "decoder.py", "kv_mask=kv_mask, kv_sort_idx=kv_sort_idx, training=training)", "kv_mask=kv_mask, training=training)"),
    (
        "m07_silu_on_x1",
        N + "dense.py",
        'activation=None if self.gate else activation, use_bias=bias, name=sub(f"hidden{i}")',
        'activation="silu" if self.gate else activation, use_bias=bias, name=sub(f"hidden{i}")',
    ),
    (
        "m08_product_not_a_leaf",
        N + "dense.py",
        "                x = self.products[i]([layer(x, training=training), self.gates[i](x, training=training)], training=training)",
        "                x = layer(x, training=training) * self.gates[i](x, training=training)",
    ),
    ("m09_no_inf_fill", N + "encoder.py", '        value = keras.ops.where(valid, value, float("inf"))\n', "        pass\n"),
    ("m10_mask_not_sorted", N + "encoder.py", "                kv_mask = gather_tokens(kv_mask, sort_idx)\n", "                pass\n"),
    ("m11_no_unsort", N + "encoder.py", "            x = gather_tokens(x, keras.ops.argsort(sort_idx, axis=-1))\n", "            pass\n"),
    (
        "m12_unsort_with_sort_idx",
        N + "encoder.py",
        "            x = gather_tokens(x, keras.ops.argsort(sort_idx, axis=-1))\n",
        "            x = gather_tokens(x, sort_idx)\n",
    ),
    ("m13_no_key_sort_idx", N + "maskformer.py", '            x["key_sort_idx"] = sort_index(sort_value, x["key_valid"])\n', ""),
    ("m14_no_dim_check", N + "maskformer.py", "        if enc_dim != dim:\n", "        if False:\n"),
    (
        "m15_b1_drops_mask",
        N + "maskformer.py",
        '        x["key_valid"] = x[f"{name}_valid"]\n',
        '        x["key_valid"] = x[f"{name}_valid"]\n        if x["key_valid"].shape[0] == 1 and bool(keras.ops.all(x["key_valid"])):\n            x["key_valid"] = None\n',
    ),
    ("m16_sort_not_applied", N + "maskformer.py", "x_sort_value=sort_value, training=training)", "training=training)"),
    (
        "m17_gate_bias_from_x1",
        "tests/keras_native/porting.py",
        "            assign_dense(ndense.gates[i], w2, b2)\n",
        "            assign_dense(ndense.gates[i], w2, b1)\n",
    ),
]
