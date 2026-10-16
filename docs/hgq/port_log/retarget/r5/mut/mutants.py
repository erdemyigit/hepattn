"""R5 mutants: (file, old, new, count) applied to an rsync copy of the worktree. Usage: mutants.py <name> <root>."""
import sys
from pathlib import Path

M = {
    "m00_none": [],
    "m01_no_kv_sort_gather": [("src/hepattn/keras_native/attention.py", "        if kv_sort_idx is not None:\n", "        if False:\n", 1)],
    "m02_tail_not_in_renorm": [("src/hepattn/keras_native/masked_linformer.py", "full2 = full2 + ops.sum(ops.square(tail), 0)", "full2 = full2", 1)],
    "m03_encoder_no_sort": [("src/hepattn/keras_native/encoder.py", "if x_sort_value is not None:", "if False:", None)],
    "m04_decoder_order_ignored": [("src/hepattn/keras_native/decoder.py", 'if self.query_update_order == "ca_sa_dense":', "if False:", 1)],
    "m05_swiglu_no_silu": [("src/hepattn/keras_native/dense.py", 'factory.dense(size, activation="silu", use_bias=bias, name=sub(f"hidden{i}_x2")', 'factory.dense(size, activation=None, use_bias=bias, name=sub(f"hidden{i}_x2")', 1)],
    "m06_project_renorm_off": [("src/hepattn/keras_native/masked_linformer.py", "if kv_mask is not None and self.renorm:", "if False:", 1)],
    "m07_old_cross_event_bce": [("src/hepattn/keras_native/losses.py", "    per_object = ops.sum(loss * ops.expand_dims(pad, 1), axis=-1) / counts\n    return _mean_over_valid_objects(per_object, object_valid_mask)\n\n\ndef mask_dice_loss",
                                 '    per_object = ops.mean(ops.einsum("qpn,bn->bqp", loss, pad) / counts[:, :, None], axis=0)\n    return _mean_over_valid_objects(per_object, object_valid_mask)\n\n\ndef mask_dice_loss', 1)],
}

name, root = sys.argv[1], Path(sys.argv[2])
for rel, old, new, count in M[name]:
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    assert n >= 1 and (count is None or n == count), (name, rel, n)
    p.write_text(s.replace(old, new))
print(name, "applied", len(M[name]))
