# ruff: noqa: E501  (mutant source strings are exact one-line matches)
N = "src/hepattn/keras_native/"
MUTANTS = {
    # reviewer's x04: host-side numpy sort; the traced-kwargs JIT gate must reject it
    "x04_numpy_sort": (
        N + "encoder.py",
        '    if valid is not None:\n        value = keras.ops.where(valid, value, float("inf"))\n    return keras.ops.argsort(value, axis=-1)\n',
        "    import numpy as np\n    value = np.asarray(value)\n    if valid is not None:\n        value = np.where(np.asarray(valid), value, np.inf)\n    return np.argsort(value, axis=-1, kind='stable').astype('int32')\n",
    ),
    # reviewer's x07: stray keys next to the encoder class_path ignored
    "x07_extra_keys_ignored": (
        N + "maskformer.py",
        "        if spec:\n            raise ValueError",
        "        if False:\n            raise ValueError",
    ),
    # the SwiGLU product keeps the "..." equation (pre-fix behaviour)
    "f01_ellipsis_equation": (N + "dense.py", "                self.products[i].equation = product_equation\n", ""),
    # SwiGLU halves get the unpacked glorot bound (pre-fix behaviour)
    "f02_unpacked_glorot": (
        N + "dense.py",
        "            return initializers(fan_ins[i], 2 * size if self.gate else None)",
        "            return initializers(fan_ins[i], None)",
    ),
    # native mask logits shifted by 1e-4: the elementwise e2e parity check already rejects it (atol 5e-5 at small logits)
    "f03_logit_shift": (
        N + "tasks.py",
        "        logit = self.logit_scale * self.mask_einsum([mask_tokens, xs], training=training)",
        "        logit = self.logit_scale * self.mask_einsum([mask_tokens, xs], training=training) + 1e-4",
    ),
    # native mask logits scaled by (1 + 4e-5): inside the e2e rtol (1e-4) everywhere, so parity passes and no boolean flips,
    # but the error near logit(0.1) (~9e-5) exceeds a tenth of layer 3's margin (5.4e-4): only the margin rule rejects it
    "f04_logit_scale": (
        N + "tasks.py",
        "        logit = self.logit_scale * self.mask_einsum([mask_tokens, xs], training=training)",
        "        logit = self.logit_scale * self.mask_einsum([mask_tokens, xs], training=training) * (1 + 4e-5)",
    ),
    # R3's original dense.py mutants, re-run on the fixed dense.py
    "m07_silu_on_x1": (
        N + "dense.py",
        'activation=None if self.gate else activation, use_bias=bias, name=sub(f"hidden{i}")',
        'activation="silu" if self.gate else activation, use_bias=bias, name=sub(f"hidden{i}")',
    ),
    "m08_product_not_a_leaf": (
        N + "dense.py",
        "                x = self.products[i]([layer(x, training=training), self.gates[i](x, training=training)], training=training)",
        "                x = layer(x, training=training) * self.gates[i](x, training=training)",
    ),
}
