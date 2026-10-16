import sys, re
name, root = sys.argv[1], sys.argv[2]
def sub(path, old, new, count=1):
    p = f"{root}/{path}"; s = open(p).read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))
L = "src/hepattn/keras_native/losses.py"
OLD_BCE_BODY = '''    pad = f32(input_pad_mask)
    counts = ops.maximum(ops.sum(pad, axis=-1, keepdims=True), 1.0)  # [B, 1]
    per_object = ops.sum(loss * ops.expand_dims(pad, 1), axis=-1) / counts
    return _mean_over_valid_objects(per_object, object_valid_mask)'''
NEW_BCE_OLDMATH = '''    pad = f32(input_pad_mask)
    term = ops.einsum("qpn,bn->bqp", loss, pad) / ops.maximum(ops.sum(pad, -1), 1.0)[:, None, None]
    valid_qp = f32(object_valid_mask)[None]
    return ops.sum(term * valid_qp) / ops.maximum(ops.cast(ops.shape(term)[0], "float32") * ops.sum(valid_qp), 1.0)'''
OLD_DICE_BODY = '''    probs = ops.sigmoid(f32(logits)) * ops.expand_dims(f32(input_pad_mask), 1)
    targets = f32(targets)
    numerator = 2.0 * ops.sum(probs * targets, axis=-1)
    denominator = ops.sum(probs, axis=-1) + ops.sum(targets, axis=-1)
    per_object = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return _mean_over_valid_objects(per_object, object_valid_mask)'''
NEW_DICE_OLDMATH = '''    probs, pad, tgt = ops.sigmoid(f32(logits)), f32(input_pad_mask), f32(targets)
    num = 2 * ops.einsum("qpn,bn->bqp", probs * tgt, pad)
    den = ops.einsum("qpn,bn->bqp", probs, pad) + ops.sum(tgt, -1)[None]
    term = 1 - (num + 1) / (den + 1)
    valid_qp = f32(object_valid_mask)[None]
    return ops.sum(term * valid_qp) / ops.maximum(ops.cast(ops.shape(term)[0], "float32") * ops.sum(valid_qp), 1.0)'''
if name == "b1":
    sub(L, OLD_BCE_BODY, NEW_BCE_OLDMATH); sub(L, OLD_DICE_BODY, NEW_DICE_OLDMATH)
elif name == "b1dice":
    sub(L, OLD_DICE_BODY, NEW_DICE_OLDMATH)
elif name == "b2":
    sub("src/hepattn/keras_native/objective.py", 'logits = outputs[self.output_object + "_logit"]\n        target = targets[self.target_object + "_class"]\n        return {name: w * L.object_ce_cost',
        'logits = outputs[self.output_object + "_class_prob"]\n        target = targets[self.target_object + "_class"]\n        return {name: w * L.object_ce_cost')
print("mutated", name)
N = "src/hepattn/keras_native/"
H = "src/hepattn/keras/"
if name == "n1":
    sub(N + "matching.py", "costs = np.nan_to_num(np.asarray(costs, dtype=np.float32), nan=_BIG, posinf=_BIG, neginf=-_BIG)", "costs = np.asarray(costs, dtype=np.float32)")
elif name == "n2":
    sub(N + "masked_linformer.py", "bound = 1.0 / math.sqrt(k)  # masked_linformer.init_", "bound = 1.0 / k  # MUTANT")
elif name == "n3":
    sub(N + "decoder.py", "        if self.self_attn_norm_kv:\n            return self.q_sa(q, q_mask=q_mask, training=training)\n", "")
elif name == "n4":
    sub(N + "norm.py", 'NON_AFFINE_NORMS = ("FastLayerNorm",)', "NON_AFFINE_NORMS = ()")
elif name == "n5":
    sub(N + "encoder.py", '        value = keras.ops.where(valid, value, float("inf"))\n', "        pass\n")
elif name == "n6":
    sub(N + "norm.py", 'if dense_norm_placement == "hybridnorm" else not is_hybrid_subsequent', 'if True else not is_hybrid_subsequent')
elif name == "n7":
    sub(N + "attention.py", "    if unknown:\n        raise TypeError(f\"unknown attention option(s)", "    if False:\n        raise TypeError(f\"unknown attention option(s)")
elif name == "h1":
    sub(H + "attention.py", "        if kv_sort_idx is not None:  # the projections index key slots", "        if False:  # MUTANT the projections index key slots")
elif name == "h2":
    sub(H + "encoder.py", "        if x_sort_value is not None:\n            x_sort_idx = sort_index", "        if False:\n            x_sort_idx = sort_index")
