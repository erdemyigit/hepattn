"""Keras-native MaskFormer (``hepattn.models.maskformer.MaskFormer.forward``) as a ``keras.Model``.

``call(inputs, training)`` returns the torch output tree: ``{"encoder": {}, "layer_{i}": {task: {...}, "attn_mask"},
"final": {task: {...}}}``. The HGQ2 regularisation terms are not walked from a module tree (hybrid
``quant_losses``): under JAX they come back from ``model.stateless_call(..., return_losses=True)``, and eagerly from
``model.losses`` after a training-mode call.

Construction: ``MaskFormer(input_nets=[...], encoder={...}, decoder={...}, tasks=[...], dim=..., quant=...)`` with
plain dicts, or ``from_config`` with the ``model:`` section of a hepattn YAML (``pflow_hgq.yaml``, or the generated
``polaris_hgq.yaml``). Every layer is created inside ``LayerFactory.scopes()``, and ``build`` runs inside them too.

Unsupported options raise: encoder tasks, pooling, sorting, unified decoding, more than one input type, and all
the decoder / encoder / task options listed in those modules. ``from_config`` also raises for ``mtl: true``.
"""

import keras

from hepattn.keras_native.decoder import MaskFormerDecoder
from hepattn.keras_native.encoder import Encoder
from hepattn.keras_native.factory import LayerFactory
from hepattn.keras_native.input import InputNet
from hepattn.keras_native.tasks import build_task, dense_kwargs

MASKFORMER_CLASSES = {"MaskFormer", "KerasMaskFormer"}


class MaskFormer(keras.Model):
    def __init__(
        self,
        input_nets: list[dict],
        encoder: dict,
        decoder: dict,
        tasks: list[dict],
        dim: int,
        target_object: str = "particle",
        matcher: dict | None = None,
        encoder_tasks: list | None = None,
        pooling: dict | None = None,
        input_sort_field: str | None = None,
        sorter: dict | None = None,
        unified_decoding: bool = False,
        dynamic_query_source: str = "hit",
        quant: dict | None = None,
        name: str = "maskformer",
        **kwargs,
    ) -> None:
        """Build the model (float with ``quant=None``, HGQ2 with a QuantSpec dict).

        Args:
            input_nets: ``InputNet`` kwargs per input (``input_name``, ``fields``, ``net``, ``posenc``); exactly one.
            encoder: ``Encoder`` kwargs without ``dim``.
            decoder: ``MaskFormerDecoder`` kwargs.
            tasks: Task specs ``{"class_path": ..., "init_args": {...}}`` (``tasks.build_task``).
            dim: Embedding dimension.
            target_object: Target object name (used by matching, S4).
            matcher: The matcher config; stored only (host matching is S4).
            encoder_tasks: Must be empty (not ported).
            pooling: Must be None (not ported).
            input_sort_field: Must be None (not ported).
            sorter: Must be None (not ported).
            unified_decoding: Must be False (not ported).
            dynamic_query_source: Ignored (dynamic queries raise in the decoder).
            quant: None, or a QuantSpec dict (weight, datalane, table, ebops).
            name: Model name.
            **kwargs: Passed to ``keras.Model``.

        Raises:
            NotImplementedError: For any unsupported option above.
            ValueError: For a reserved input name ("key", "query") or one with an underscore.
        """
        super().__init__(name=name, **kwargs)
        for flag, what in (
            (bool(encoder_tasks), "encoder_tasks"),
            (pooling is not None, "pooling"),
            (input_sort_field is not None or sorter is not None, "input sorting"),
            (unified_decoding, "unified_decoding"),
            (len(input_nets) != 1, f"{len(input_nets)} input types (exactly one is ported)"),
        ):
            if flag:
                raise NotImplementedError(f"{what} is not ported to keras_native")
        del dynamic_query_source
        self.factory = LayerFactory(quant)
        self.dim = dim
        self.target_object = target_object
        self.matcher_config = matcher
        with self.factory.scopes():
            self.input_nets = [InputNet(factory=self.factory, name=f"innet{i}", **spec) for i, spec in enumerate(input_nets)]
            self.encoder = Encoder(dim=dim, factory=self.factory, **encoder)
            self.decoder = MaskFormerDecoder(factory=self.factory, **decoder)
            self.tasks = [build_task(spec, self.factory, layer_name=f"task{i}") for i, spec in enumerate(tasks)]
        self.decoder.set_tasks(self.tasks)
        self.input_name = self.input_nets[0].input_name
        for reserved in ("key", "query"):
            if self.input_name == reserved:
                raise ValueError(f"'{reserved}' input name is reserved.")
        if "_" in self.input_name:
            raise ValueError("Input names cannot contain underscores.")

    def build(self, inputs_shape) -> None:
        name = self.input_name
        num = inputs_shape[f"{name}_valid"][1]
        embed = (None, num, self.dim)
        with self.factory.scopes():
            for net in self.input_nets:
                net.build(inputs_shape)
            self.encoder.build(embed)
            self.decoder.build(embed)
            x_shape = {
                "inputs": inputs_shape,
                "query_embed": (None, self.decoder.num_queries, self.dim),
                f"{name}_embed": embed,
                "key_embed": embed,
                f"{name}_valid": inputs_shape[f"{name}_valid"],
            }
            for task in self.tasks:
                task.build(x_shape)

    def call(self, inputs: dict, training=None, attn_masks: dict | None = None):
        """The torch output tree. ``attn_masks`` forces decoder attention masks per layer (see ``MaskFormerDecoder.call``)."""
        name = self.input_name
        x = {"inputs": inputs}
        x[f"{name}_embed"] = self.input_nets[0](inputs, training=training)
        x[f"{name}_valid"] = inputs[f"{name}_valid"]
        # one input type: the merged keys are its constituents (torch concatenates over input types)
        x["key_valid"] = x[f"{name}_valid"]
        x["key_embed"] = self.encoder(x[f"{name}_embed"], kv_mask=x["key_valid"], training=training)
        x[f"{name}_embed"] = x["key_embed"]

        x, layer_outputs = self.decoder(x, input_name=name, attn_masks=attn_masks, training=training)
        outputs = {"encoder": {}, **layer_outputs}
        final = {}
        for task in self.tasks:
            final[task.task_name] = task(x, outputs=final, training=training)
        outputs["final"] = final
        return outputs


def _modules(spec) -> list:
    """The module list of ``{"class_path": "torch.nn.ModuleList", "init_args": {"modules": [...]}}`` or a plain list.

    Raises:
        ValueError: For a dict spec whose class is not a ModuleList.
    """
    if spec is None:
        return []
    if isinstance(spec, dict):
        if spec.get("class_path", "").rsplit(".", 1)[-1] != "ModuleList":
            raise ValueError(f"expected a ModuleList spec, got {spec.get('class_path')}")
        return list((spec.get("init_args") or {}).get("modules") or [])
    return list(spec)


def _input_net_kwargs(spec: dict) -> dict:
    class_path = spec.get("class_path", "")
    if class_path.rsplit(".", 1)[-1] != "InputNet":
        raise NotImplementedError(f"input net '{class_path}' is not ported (only hepattn InputNet)")
    args = dict(spec.get("init_args") or {})
    args["net"] = dense_kwargs(args["net"])
    posenc = args.get("posenc")
    if posenc is not None and "class_path" in posenc:
        args["posenc"] = {"class_path": posenc["class_path"], **(posenc.get("init_args") or {})}
    return args


def from_config(config: dict) -> MaskFormer:
    """A native MaskFormer from the ``model:`` section of a hepattn CLIC YAML.

    Accepts the whole section (``{"mtl": ..., "optimizer": ..., "model": {"class_path": ..., "init_args": ...}}``),
    the ``model`` entry, or its ``init_args``. Optimizer and LR settings are not read here (S5).

    Raises:
        NotImplementedError: For ``mtl: true``, a model class other than (Keras)MaskFormer, or any option the
            native layers do not port.
    """
    config = dict(config)
    if config.get("mtl"):
        raise NotImplementedError("mtl (multi-task gradient balancing) is not ported to keras_native")
    if "model" in config:
        config = dict(config["model"])
    if "class_path" in config:
        if config["class_path"].rsplit(".", 1)[-1] not in MASKFORMER_CLASSES:
            raise NotImplementedError(f"model '{config['class_path']}' is not a MaskFormer")
        config = dict(config.get("init_args") or {})
    config["input_nets"] = [_input_net_kwargs(spec) for spec in _modules(config["input_nets"])]
    config["tasks"] = _modules(config["tasks"])
    config["encoder_tasks"] = _modules(config.get("encoder_tasks"))
    return MaskFormer(**config)
