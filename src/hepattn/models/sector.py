"""Sectored MaskFormer: the full-event encoder, then one decoder pass per phi sector of each event.

The nodes of each event are split into S sectors of at most M nodes (for CLIC, `experiments/clic/sectors.py`); the
inputs `<input>_sector` and `<input>_sector_rank` give each node's sector and its position in the sector. The decoder
runs on B x S rows of M node slots (in rank order, padding last) with Q learned queries shared by all rows. Its outputs
are returned in the event frame: sector j owns the object slots [j Q, (j + 1) Q), and a node-axis output of an object
is filled outside the object's sector (mask logit: finfo.min, incidence: 0, attention mask: False).

Targets must hold S blocks of Q object slots (valid targets first in each block, block j for sector j), with mask and
incidence targets inside the block's sector, plus `<input>_sector` and `<input>_sector_rank` as in the inputs.
Matching is per block. `loss_norm` sets how the losses are normalised:
- "sector_mean": the losses of each sector, with the reference formulas on the sector's rows (Q slots, M nodes),
  averaged over the S sectors with weight 1 / S. A sector index is pooled over the batch as the reference pools
  over events, so S = 1 gives the reference loss and batch size 1 gives the per-event sector average of HEPTv2
  (arXiv:2606.20437, Sec. 4.2.3). A loss term without valid targets in a sector counts as 0 in that sector's share.
- "event": the reference losses and costs on the event-frame outputs.
"""

import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.maskformer import MaskFormer
from hepattn.models.task import (
    IncidenceBasedRegressionTask,
    IncidenceRegressionTask,
    ObjectClassificationTask,
    ObjectHitMaskTask,
    ObjectRegressionTask,
)

LOSS_NORMS = ("sector_mean", "event")
FINFO_MIN = "finfo_min"


def row_index(node_sector: Tensor, node_sector_rank: Tensor, node_valid: Tensor, num_sectors: int, max_nodes: int) -> tuple[Tensor, Tensor]:
    """Event node index of every row slot.

    Args:
        node_sector: (B, N) sector of each node; ignored on padding.
        node_sector_rank: (B, N) position of each node in its sector, 0 .. size - 1.
        node_valid: (B, N) True for real nodes.
        num_sectors: S.
        max_nodes: M, slots per row.

    Returns:
        index: (B, S, M) event node index of each slot (0 on empty slots).
        row_valid: (B * S, M) True on slots that hold a node; real nodes come first in every row.

    Raises:
        ValueError: If a valid node has a sector or rank outside [0, S) x [0, M).
    """
    batch_size, num_nodes = node_sector.shape
    bad = node_valid & ((node_sector < 0) | (node_sector >= num_sectors) | (node_sector_rank < 0) | (node_sector_rank >= max_nodes))
    if bad.any():
        raise ValueError(f"node sector/rank outside {num_sectors} sectors x {max_nodes} slots")
    dump = num_sectors * max_nodes
    slot = torch.where(node_valid, node_sector * max_nodes + node_sector_rank, dump)
    nodes = torch.arange(num_nodes, device=slot.device).expand(batch_size, -1)
    index = torch.zeros(batch_size, dump + 1, dtype=torch.long, device=slot.device).scatter_(1, slot, nodes)
    valid = torch.zeros(batch_size, dump + 1, dtype=torch.bool, device=slot.device).scatter_(1, slot, node_valid)
    return index[:, :dump].view(batch_size, num_sectors, max_nodes), valid[:, :dump].reshape(batch_size * num_sectors, max_nodes)


def _fill(t: Tensor, fill: float | str) -> Tensor:
    value = torch.finfo(t.dtype).min if fill == FINFO_MIN else fill
    return torch.full((), value, dtype=t.dtype, device=t.device)


def gather_rows(t: Tensor, index: Tensor, row_valid: Tensor, fill: float | str = 0) -> Tensor:
    """(B, N, ...) per-node tensor -> (B * S, M, ...) rows; empty slots hold `fill`."""
    batch_size, num_sectors, max_nodes = index.shape
    rest = t.shape[2:]
    idx = index.reshape(batch_size, num_sectors * max_nodes, *(1,) * len(rest)).expand(-1, -1, *rest)
    rows = t.gather(1, idx).reshape(batch_size * num_sectors, max_nodes, *rest)
    return torch.where(row_valid.view(*row_valid.shape, *(1,) * len(rest)), rows, _fill(t, fill))


def scatter_rows(t: Tensor, index: Tensor, row_valid: Tensor, num_nodes: int, fill: float | str = 0) -> Tensor:
    """(B * S, Q, M) row output with a node axis last -> (B, S * Q, N); entries outside an object's sector hold `fill`."""
    batch_size, num_sectors, max_nodes = index.shape
    num_queries = t.shape[1]
    dest = torch.where(row_valid.view(batch_size, num_sectors, max_nodes), index, num_nodes)
    dest = dest[:, :, None, :].expand(-1, -1, num_queries, -1)
    out = torch.full((batch_size, num_sectors, num_queries, num_nodes + 1), _fill(t, fill).item(), dtype=t.dtype, device=t.device)
    out = out.scatter(3, dest, t.view(batch_size, num_sectors, num_queries, max_nodes))
    return out[..., :num_nodes].reshape(batch_size, num_sectors * num_queries, num_nodes)


def gather_node_axis(t: Tensor, index: Tensor, row_valid: Tensor, fill: float | str = 0) -> Tensor:
    """(B, S * Q, N) event-frame tensor with a node axis -> (B * S, Q, M): the inverse of `scatter_rows` on the rows."""
    batch_size, num_sectors, max_nodes = index.shape
    num_queries = t.shape[1] // num_sectors
    rows = t.view(batch_size, num_sectors, num_queries, -1).gather(3, index[:, :, None, :].expand(-1, -1, num_queries, -1))
    rows = torch.where(row_valid.view(batch_size, num_sectors, 1, max_nodes), rows, _fill(t, fill))
    return rows.reshape(batch_size * num_sectors, num_queries, max_nodes)


def block_match(matcher: nn.Module, costs: Tensor, target_valid: Tensor, num_blocks: int) -> Tensor:
    """Assignment restricted to the diagonal blocks of the cost matrix.

    Args:
        matcher: Called as matcher(costs (n, Q, Q), target_valid (n, Q), None) -> (n, Q) prediction indices.
        costs: (n, S * Q, S * Q) costs, axes (batch, pred, true).
        target_valid: (n, S * Q) valid targets, a prefix of every block.
        num_blocks: S.

    Returns:
        (n, S * Q) prediction index for every target slot, always inside the slot's block.
    """
    n, num_pred, _ = costs.shape
    num_queries = num_pred // num_blocks
    blocks = costs.view(n, num_blocks, num_queries, num_blocks, num_queries).diagonal(dim1=1, dim2=3)  # (n, Q, Q, S)
    blocks = blocks.permute(0, 3, 1, 2).reshape(n * num_blocks, num_queries, num_queries)
    idx = matcher(blocks, target_valid.reshape(n * num_blocks, num_queries), None).view(n, num_blocks, num_queries)
    offset = num_queries * torch.arange(num_blocks, device=idx.device)[:, None]
    return (idx + offset).reshape(n, num_pred)


def _node_axis_fills(task: nn.Module) -> dict[str, float | str]:
    """Outputs of a task with a node axis, and their fill outside an object's sector.

    Raises:
        ValueError: For a task type whose outputs are not known here.
    """
    if isinstance(task, ObjectHitMaskTask):
        return {task.output_object_hit + "_logit": FINFO_MIN}
    if isinstance(task, IncidenceRegressionTask):
        return {task.incidence_key: 0.0}
    if isinstance(task, ObjectClassificationTask | IncidenceBasedRegressionTask | ObjectRegressionTask):
        return {}
    raise ValueError(f"SectoredMaskFormer does not know the output axes of {type(task).__name__}")


class SectoredMaskFormer(MaskFormer):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        decoder: MaskFormerDecoder,
        tasks: nn.ModuleList,
        dim: int,
        num_sectors: int,
        max_nodes_per_sector: int,
        loss_norm: str = "sector_mean",
        target_object: str = "particle",
        matcher: nn.Module | None = None,
        input_sort_field: str | None = None,
        encoder_tasks: nn.ModuleList | None = None,
    ):
        """MaskFormer whose decoder runs per phi sector; see the module docstring.

        Args:
            input_nets: As MaskFormer; exactly one input type.
            encoder: As MaskFormer.
            decoder: As MaskFormer, with `num_queries` = Q per sector and Linformer `linformer_seq_len` >= max(M, Q).
            tasks: As MaskFormer; object-level tasks plus ObjectHitMaskTask and IncidenceRegressionTask.
            dim: As MaskFormer.
            num_sectors: S.
            max_nodes_per_sector: M, node slots per decoder row.
            loss_norm: "sector_mean" (per-sector losses averaged with 1 / S) or "event" (event-frame losses).
            target_object: As MaskFormer.
            matcher: As MaskFormer.
            input_sort_field: As MaskFormer; sorts the encoder input only.
            encoder_tasks: As MaskFormer; they run on the full event.

        Raises:
            ValueError: For an unknown loss_norm, a task with unknown output axes, or a decoder option the rows do
                not support (dynamic queries, positional encodings, local attention).
        """
        super().__init__(
            input_nets=input_nets,
            encoder=encoder,
            decoder=decoder,
            tasks=tasks,
            dim=dim,
            target_object=target_object,
            matcher=matcher,
            input_sort_field=input_sort_field,
            encoder_tasks=encoder_tasks,
        )
        if loss_norm not in LOSS_NORMS:
            raise ValueError(f"loss_norm must be one of {LOSS_NORMS}, got {loss_norm}")
        if len(self.input_names) != 1:
            raise ValueError("SectoredMaskFormer supports one input type")
        if decoder.dynamic_queries or decoder.posenc or decoder.local_strided_attn:
            raise ValueError("SectoredMaskFormer does not support dynamic queries, positional encodings or local attention")
        self.num_sectors = num_sectors
        self.max_nodes_per_sector = max_nodes_per_sector
        self.loss_norm = loss_norm
        self.node_axis_fills = {task.name: _node_axis_fills(task) for task in tasks}

    @property
    def num_queries(self) -> int:
        return self.decoder.num_queries({})

    def rows(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Row index and row validity from the `<input>_sector`, `<input>_sector_rank` and `<input>_valid` fields."""
        name = self.input_names[0]
        return row_index(data[f"{name}_sector"], data[f"{name}_sector_rank"], data[f"{name}_valid"], self.num_sectors, self.max_nodes_per_sector)

    def decode(self, x: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        name = self.input_names[0]
        index, row_valid = self.rows(x["inputs"])
        batch_size, num_nodes = x[f"{name}_valid"].shape
        embed = gather_rows(x["key_embed"], index, row_valid)
        # The rows are in rank order already: the encoder's key_sort_idx indexes the event and is not passed on
        rows = {
            "inputs": {
                k: gather_rows(v, index, row_valid)
                for k, v in x["inputs"].items()
                if k.startswith(f"{name}_") and v.shape[:2] == (batch_size, num_nodes)
            },
            "key_embed": embed,
            f"{name}_embed": embed,
            "key_valid": row_valid,
            "key_valid_full": row_valid,
            f"{name}_valid": row_valid,
            f"key_is_{name}": torch.ones_like(row_valid),
        }
        return self.decoder(rows, self.input_names)

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, dict[str, dict[str, Tensor]]]:
        index, row_valid = self.rows(inputs)
        num_nodes = inputs[f"{self.input_names[0]}_valid"].shape[1]
        outputs = super().forward(inputs)
        return {k: v if k == "encoder" else self._layer_to_event(v, index, row_valid, num_nodes) for k, v in outputs.items()}

    def _layer_to_event(self, layer: dict, index: Tensor, row_valid: Tensor, num_nodes: int) -> dict:
        batch_size = index.shape[0]
        out = {}
        for key, value in layer.items():
            if key == "attn_mask":
                out[key] = scatter_rows(value, index, row_valid, num_nodes, fill=False)
            elif key in self.node_axis_fills:
                fills = self.node_axis_fills[key]
                out[key] = {
                    k: scatter_rows(v, index, row_valid, num_nodes, fills[k]) if k in fills else v.reshape(batch_size, -1, *v.shape[2:])
                    for k, v in value.items()
                }
            else:
                raise ValueError(f"unexpected decoder output {key}")
        return out

    def _layer_to_rows(self, layer: dict, index: Tensor, row_valid: Tensor) -> dict:
        rows_count = row_valid.shape[0]
        out = {}
        for key, value in layer.items():
            if key not in self.node_axis_fills:
                continue
            fills = self.node_axis_fills[key]
            out[key] = {
                k: gather_node_axis(v, index, row_valid, fills[k]) if k in fills else v.reshape(rows_count, -1, *v.shape[2:])
                for k, v in value.items()
            }
        return out

    def _targets_to_rows(self, targets: dict, index: Tensor, row_valid: Tensor) -> dict:
        name = self.input_names[0]
        batch_size, num_sectors, _ = index.shape
        num_nodes = targets[f"{name}_valid"].shape[1]
        num_slots = num_sectors * self.num_queries
        rows = {f"{name}_valid": row_valid}
        for key, t in targets.items():
            if not key.startswith(f"{self.target_object}_") or t.dim() < 2 or t.shape[1] != num_slots:
                continue
            if t.dim() == 3 and t.shape[2] == num_nodes:
                rows[key] = gather_node_axis(t, index, row_valid, fill=0)
            else:
                rows[key] = t.reshape(batch_size * num_sectors, self.num_queries, *t.shape[2:])
        return rows

    def solve_matching(self, costs: Tensor, target_valid: Tensor, query_valid: Tensor | None) -> Tensor:
        """Event-frame costs: solve on the diagonal (sector) blocks. Row costs (one block per row): solve as is.

        Raises:
            ValueError: If query masks are given.
        """
        if query_valid is not None:
            raise ValueError("SectoredMaskFormer does not support query masks")
        if costs.shape[1] == self.num_queries:
            return self.matcher(costs, target_valid, None)
        return block_match(self.matcher, costs, target_valid, self.num_sectors)

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        if self.loss_norm == "event":
            return super().loss(outputs, targets)
        targets, encoder_outputs, decoder_outputs = self._prepare_targets_and_outputs(outputs, targets)
        losses = self._compute_encoder_losses(encoder_outputs, targets)

        index, row_valid = self.rows(targets)
        num_nodes = targets[f"{self.input_names[0]}_valid"].shape[1]
        row_outputs = {k: self._layer_to_rows(v, index, row_valid) for k, v in decoder_outputs.items()}
        row_targets = self._targets_to_rows(targets, index, row_valid)
        costs = self._compute_decoder_costs(row_outputs, row_targets)
        self._match_and_permute_outputs(row_outputs, costs, row_targets)
        losses.update(self._sector_mean_losses(row_outputs, row_targets))

        # Matched outputs back to the event frame, aligned with the event-frame targets
        for layer_name, layer in row_outputs.items():
            outputs[layer_name].update(self._layer_to_event(layer, index, row_valid, num_nodes))
        return outputs, targets, losses

    def _sector_mean_losses(self, row_outputs: dict, row_targets: dict) -> dict[str, dict[str, dict[str, Tensor]]]:
        num_sectors = self.num_sectors

        def sector(tree, j):
            if isinstance(tree, dict):
                return {k: sector(v, j) for k, v in tree.items()}
            return tree.view(-1, num_sectors, *tree.shape[1:])[:, j]

        valid = row_targets[f"{self.target_object}_valid"].view(-1, num_sectors, self.num_queries)
        has_targets = valid.any(-1).any(0).tolist()
        total: dict[str, dict[str, dict[str, Tensor]]] = {}
        for j in range(num_sectors):
            sector_losses = self._compute_decoder_losses(sector(row_outputs, j), sector(row_targets, j))
            for layer_name, layer_losses in sector_losses.items():
                for task_name, task_losses in layer_losses.items():
                    acc = total.setdefault(layer_name, {}).setdefault(task_name, {})
                    for loss_name, value in task_losses.items():
                        # A term over valid targets is undefined (nan) in a sector without targets: it adds 0
                        term = value if has_targets[j] else torch.where(value.isnan(), 0.0, value)
                        acc[loss_name] = acc.get(loss_name, 0.0) + term / num_sectors
        return total
