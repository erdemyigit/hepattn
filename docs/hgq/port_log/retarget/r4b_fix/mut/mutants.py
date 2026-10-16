"""R4b fix-round mutants: (file, old, new). Applied to scratch copies only."""
A = "src/hepattn/keras/attention.py"
E = "src/hepattn/keras/evaluate.py"
CB = "src/hepattn/keras/callbacks.py"
MUTANTS = {
    "f01_no_save_guard": (A, '        self._require_core("state_dict()")\n', "        pass\n"),
    "f02_no_load_guard": (A, '        self._require_core("load_state_dict()")\n', "        pass\n"),
    "f03_materialize_no_forward": ("src/hepattn/keras/maskformer.py", "        self.eval()\n        self(inputs)\n", "        self.eval()\n"),
    "f04_evaluate_total_top_level": (E, "    return leaves_total_ebops(model)\n", "    return sum(float(torch.as_tensor(layer.ebops)) for layer in model.keras_layers() if getattr(layer, 'ebops', None) is not None)\n"),
    "f05_effective_beta_top_level": (E, '    layers = [layer for layer in quantized_leaves(model) if getattr(layer, "_beta", None) is not None]', '    layers = [layer for layer in model.keras_layers() if getattr(layer, "_beta", None) is not None]'),
    "f06_region_first_path": (E, "if best is not None and (not best[0] or len(path) < len(best[0])):", "if best is not None and not best[0]:"),
    "f07_floor_from_probes": (E, "    return (q2 - q1) / (b2 - b1), floor\n", "    return (q2 - q1) / (b2 - b1), q1 - (q2 - q1) / (b2 - b1) * b1\n"),
    "f08_attend_core_eval_mode": (A, "            return self.core(q, k, v, attn_mask, training=self.training)", "            return self.core(q, k, v, attn_mask, training=False)"),
    "f09_project_core_eval_mode": (A, "        return self.core(q, k, v, kv_mask, training=self.training)", "        return self.core(q, k, v, kv_mask, training=False)"),
    "f10_write_beta_top_level": (CB, "def write_beta(model, beta: float) -> None:\n    for layer in quantized_leaves(model):", "def write_beta(model, beta: float) -> None:\n    for layer in model.keras_layers():"),
    "m15_monitor_reads_ebops_property": (CB, '    ebops = own.value if own is not None else getattr(layer, "ebops", None)', '    ebops = getattr(layer, "ebops", None)'),
}
