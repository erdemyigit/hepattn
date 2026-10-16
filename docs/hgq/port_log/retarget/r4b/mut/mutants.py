"""R4b mutants: (file, old, new). Applied to scratch copies only."""
A = "src/hepattn/keras/attention.py"
MUTANTS = {
    "m01_no_sort_gather": (A, "if kv_sort_idx is not None:  # the projections", "if False:  # the projections"),
    "m02_E_F_swapped": (A, '(("k", proj_k, self.core.E), ("v", proj_v, self.core.F))', '(("k", proj_v, self.core.E), ("v", proj_k, self.core.F))'),
    "m03_dead_rows_not_ported": (A, 'getattr(self, f"proj_{part}_dead").copy_(proj[self.core.n :])', "pass"),
    "m04_q_sa_auto_core": ("src/hepattn/keras/decoder.py", 'kwargs["linformer_core"] = "project_unmasked"', "pass"),
    "m05_core_outside_module_tree": (A, "        self.core = core\n", '        self.__dict__["core"] = core\n'),
    "m06_core_imports_jax": ("src/hepattn/keras_native/masked_linformer.py", "import math\n", "import math\n\nimport jax  # noqa\n"),
    "m07_kv_mask_not_merged": (A, "                attn_mask = attn_mask & kv_mask[:, None]\n", "                pass\n"),
    "m08_core_built_outside_scopes": (A, "        with factory.scopes():\n            if kind ==", "        with __import__('contextlib').nullcontext():\n            if kind =="),
    "m09_no_core_copy_in_warm_start": ("src/hepattn/keras/porting.py", "dst_modules[path].load_projections(*module.full_projections())", "pass"),
    "m10_auto_always_attend": (A, 'kind = "project" if attn_mask is None else "attend"', 'kind = "attend"'),
    "m11_legacy_proj_dim_dropped": (A, '        raise TypeError("linformer_proj_dim is not an org/main Attention option; the projected length is linformer_k (org PR #9)")', '        attn_kwargs.pop("linformer_proj_dim")'),
    "m12_no_dead_buffers": (A, "        if core.E.tail is None and kv_len < self.linformer_seq_len:", "        if False:"),
}
MUTANTS["m09_no_core_copy_in_warm_start"] = ("src/hepattn/keras/porting.py", "    shared = {key: value for key, value in src_sd.items() if key in dst_sd and dst_sd[key].shape == value.shape}", "    shared = {key: value for key, value in src_sd.items() if key in dst_sd and dst_sd[key].shape == value.shape and '_core' not in key and 'dead' not in key}")
MUTANTS["m13_callbacks_top_level_walk"] = ("src/hepattn/keras/callbacks.py", '    walk = getattr(model, "quantized_leaves", None)', "    walk = None")
MUTANTS["m14_leaves_top_level_only"] = ("src/hepattn/keras/maskformer.py", "        for module in self.modules():\n            if getattr(module, \"_beta\", None) is not None and id(module) not in seen:", "        for module in self.keras_layers():\n            if getattr(module, \"_beta\", None) is not None and id(module) not in seen:")
MUTANTS["m15_monitor_reads_ebops_property"] = ("src/hepattn/keras/callbacks.py", '        ebops = own.value if own is not None else getattr(layer, "ebops", None)', '        ebops = getattr(layer, "ebops", None)')
