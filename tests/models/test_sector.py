"""Sectored decoder: row utilities, forward, block matching and loss normalisation (issue #20)."""

import copy
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from jsonargparse import ArgumentParser
from scipy.optimize import linear_sum_assignment

from hepattn.experiments.clic.sectors import partition
from hepattn.models import MaskFormer
from hepattn.models.matcher import Matcher
from hepattn.models.sector import SectoredMaskFormer, block_match, gather_node_axis, gather_rows, row_index, scatter_rows

ROOT = Path(__file__).resolve().parents[2]
CLIC = ROOT / "src/hepattn/experiments/clic"
BASE_CONFIG = CLIC / "studies/linformer_dim/configs/small_linformer_lr6p2e-4_wd0p1_ep400_deck16.yaml"
NUM_CLASSES = 5
FINFO_MIN = torch.finfo(torch.float32).min


def model_config(num_sectors=None, max_nodes=None, num_queries=150, seq_len=160, loss_norm="sector_mean", layers=(2, 2)):
    cfg = copy.deepcopy(yaml.safe_load(BASE_CONFIG.read_text())["model"]["model"])
    args = cfg["init_args"]
    args["encoder"]["init_args"]["num_layers"] = layers[0]
    args["decoder"]["num_decoder_layers"] = layers[1]
    args["decoder"]["num_queries"] = num_queries
    args["decoder"]["decoder_layer_config"]["attn_kwargs"]["linformer_seq_len"] = seq_len
    args["matcher"]["init_args"] = {"default_solver": "scipy", "adaptive_solver": False}
    for task in args["tasks"]["init_args"]["modules"]:
        if "scale_dict_path" in task["init_args"]:
            task["init_args"]["scale_dict_path"] = str(CLIC / "configs/clic_var_transform.yaml")
    if num_sectors is not None:
        cfg["class_path"] = "hepattn.models.sector.SectoredMaskFormer"
        args.update(num_sectors=num_sectors, max_nodes_per_sector=max_nodes, loss_norm=loss_norm)
    return cfg


def build(cfg, seed=0):
    torch.manual_seed(seed)
    parser = ArgumentParser()
    parser.add_argument("--m", type=MaskFormer)
    return parser.instantiate_classes(parser.parse_object({"m": cfg})).m.eval()


def make_inputs(num_nodes, num_sectors, max_nodes, seed=0, max_len=160):
    """CLIC-like inputs (as the dummy dataset), with the `cap` sectors of the node phi."""
    gen = torch.Generator().manual_seed(seed)
    batch = len(num_nodes)
    inputs = {
        "node_features": torch.randn(batch, max_len, 27, generator=gen),
        "node_valid": torch.zeros(batch, max_len, dtype=torch.bool),
        "node_e": torch.rand(batch, max_len, generator=gen) * 100,
        "node_pt": torch.rand(batch, max_len, generator=gen) * 50,
        "node_eta": torch.randn(batch, max_len, generator=gen),
        "node_phi": (torch.rand(batch, max_len, generator=gen) * 2 - 1) * torch.pi,
        "node_is_track": (torch.rand(batch, max_len, generator=gen) < 0.4).float(),
        "node_sector": torch.full((batch, max_len), -1, dtype=torch.long),
        "node_sector_rank": torch.zeros(batch, max_len, dtype=torch.long),
    }
    inputs["node_sinphi"], inputs["node_cosphi"] = inputs["node_phi"].sin(), inputs["node_phi"].cos()
    for b, n in enumerate(num_nodes):
        inputs["node_valid"][b, :n] = True
        sector, rank, _ = partition(inputs["node_phi"][b, :n].double().numpy(), num_sectors, max_nodes)
        inputs["node_sector"][b, :n] = torch.from_numpy(sector)
        inputs["node_sector_rank"][b, :n] = torch.from_numpy(rank)
    return inputs


def make_targets(inputs, num_sectors, num_queries, targets_per_sector, seed=0):
    """Block targets of option (a): sector j's targets in slots [j Q, j Q + n), masks inside sector j."""
    gen = torch.Generator().manual_seed(seed)
    batch, max_len = inputs["node_valid"].shape
    slots = num_sectors * num_queries
    t = {
        "node_valid": inputs["node_valid"].clone(),
        "node_sector": inputs["node_sector"].clone(),
        "node_sector_rank": inputs["node_sector_rank"].clone(),
        "particle_valid": torch.zeros(batch, slots, dtype=torch.bool),
        "particle_class": torch.full((batch, slots), NUM_CLASSES, dtype=torch.long),
        "particle_node_valid": torch.zeros(batch, slots, max_len, dtype=torch.bool),
        "particle_incidence": torch.zeros(batch, slots, max_len),
    }
    for field in ("e", "pt", "eta", "sinphi", "cosphi"):
        t[f"particle_{field}"] = torch.full((batch, slots), torch.nan)
    for b in range(batch):
        for j in range(num_sectors):
            nodes = torch.nonzero(inputs["node_valid"][b] & (inputs["node_sector"][b] == j)).flatten()
            n = min(targets_per_sector[b][j], len(nodes))
            if n == 0:
                continue
            first = j * num_queries
            t["particle_valid"][b, first : first + n] = True
            t["particle_class"][b, first : first + n] = torch.randint(0, NUM_CLASSES, (n,), generator=gen)
            for field in ("e", "pt", "eta", "sinphi", "cosphi"):
                t[f"particle_{field}"][b, first : first + n] = torch.randn(n, generator=gen)
            # every target owns at least one node; the rest go to a random target or to none
            owner = torch.randint(-1, n, (len(nodes),), generator=gen)
            owner[:n] = torch.arange(n)
            for node, o in zip(nodes.tolist(), owner.tolist(), strict=True):
                if o >= 0:
                    t["particle_node_valid"][b, first + o, node] = True
                    t["particle_incidence"][b, first + o, node] = 1.0
    return t


def task_outputs(outputs):
    return {
        (layer, task, key): v
        for layer, lo in outputs.items()
        if layer != "encoder"
        for task, to in lo.items()
        if task != "attn_mask"
        for key, v in to.items()
    }


def pair(num_sectors, max_nodes, num_queries, seq_len, loss_norm="sector_mean", seed=0):
    """Reference and sectored model with the same weights."""
    ref = build(model_config(num_queries=num_queries, seq_len=seq_len), seed)
    sec = build(model_config(num_sectors, max_nodes, num_queries, seq_len, loss_norm), seed + 1)
    sec.load_state_dict(ref.state_dict())
    return ref, sec


# ---------------------------------------------------------------------------------------------------------------------
# Step 3: rows


def test_rows_in_rank_order_with_padding_last():
    inputs = make_inputs([159, 60, 2], 3, 53)
    index, row_valid = row_index(inputs["node_sector"], inputs["node_sector_rank"], inputs["node_valid"], 3, 53)
    for b in range(3):
        for j in range(3):
            members = inputs["node_valid"][b] & (inputs["node_sector"][b] == j)
            n = int(members.sum())
            valid = row_valid[b * 3 + j]
            assert valid[:n].all()
            assert not valid[n:].any()
            nodes = index[b, j, :n]
            assert (inputs["node_sector"][b, nodes] == j).all()
            assert (inputs["node_sector_rank"][b, nodes] == torch.arange(n)).all()


def test_each_node_in_one_slot():
    inputs = make_inputs([159, 60, 100], 3, 53)
    index, row_valid = row_index(inputs["node_sector"], inputs["node_sector_rank"], inputs["node_valid"], 3, 53)
    for b in range(3):
        slots = index[b].reshape(-1)[row_valid.view(3, -1)[b]]
        assert sorted(slots.tolist()) == torch.nonzero(inputs["node_valid"][b]).flatten().tolist()


def test_gather_scatter_round_trip():
    num_sectors, max_nodes, num_queries = 3, 53, 4
    inputs = make_inputs([159, 60, 100], num_sectors, max_nodes)
    index, row_valid = row_index(inputs["node_sector"], inputs["node_sector_rank"], inputs["node_valid"], num_sectors, max_nodes)
    # per-node fields: row slot (b, j, r) holds the node of sector j and rank r, empty slots hold the fill
    rows = gather_rows(inputs["node_e"], index, row_valid, fill=-7.0).view(3, num_sectors, max_nodes)
    for b, j in itertools.product(range(3), range(num_sectors)):
        for r in range(max_nodes):
            node = torch.nonzero(inputs["node_valid"][b] & (inputs["node_sector"][b] == j) & (inputs["node_sector_rank"][b] == r)).flatten()
            assert rows[b, j, r] == (inputs["node_e"][b, node[0]] if len(node) else -7.0)
    # node-axis outputs: scatter then gather is the identity on the rows; the event frame holds the fill elsewhere
    row_out = torch.randn(3 * num_sectors, num_queries, max_nodes)
    row_out = torch.where(row_valid[:, None], row_out, FINFO_MIN)
    event = scatter_rows(row_out, index, row_valid, 160, fill="finfo_min")
    assert torch.equal(gather_node_axis(event, index, row_valid, fill="finfo_min"), row_out)
    owner = torch.arange(num_sectors).repeat_interleave(num_queries)
    outside = (inputs["node_sector"][:, None, :] != owner[None, :, None]) | ~inputs["node_valid"][:, None, :]
    assert (event[outside] == FINFO_MIN).all()
    assert (event[~outside] != FINFO_MIN).all()


# ---------------------------------------------------------------------------------------------------------------------
# Step 4: sectored forward


def test_one_sector_reproduces_maskformer():
    ref, sec = pair(1, 160, 150, 160)
    inputs = make_inputs([159, 40, 97], 1, 160)
    with torch.no_grad():
        out_ref, out_sec = ref(inputs), sec(inputs)
    a, b = task_outputs(out_ref), task_outputs(out_sec)
    assert a.keys() == b.keys()
    assert len(a) >= 10
    for key in a:
        torch.testing.assert_close(b[key], a[key], rtol=1e-5, atol=1e-5, msg=str(key))
    valid = inputs["node_valid"][:, None, :]
    for layer in out_ref:
        if "attn_mask" in out_ref[layer]:
            assert torch.equal(out_ref[layer]["attn_mask"] & valid, out_sec[layer]["attn_mask"] & valid)


def test_sectors_match_an_independent_loop():
    num_sectors, max_nodes, num_queries = 3, 53, 50
    sec = build(model_config(num_sectors, max_nodes, num_queries, 53))
    inputs = make_inputs([159, 70], num_sectors, max_nodes, seed=3)
    encoded = {}
    sec.encoder.register_forward_hook(lambda _m, _i, out: encoded.update(key_embed=out))
    with torch.no_grad():
        outputs = sec(inputs)
        for b, j in itertools.product(range(2), range(num_sectors)):
            members = torch.nonzero(inputs["node_valid"][b] & (inputs["node_sector"][b] == j)).flatten()
            nodes = members[torch.argsort(inputs["node_sector_rank"][b, members])]
            sub_inputs = {k: v[b : b + 1, nodes] for k, v in inputs.items()}
            embed = encoded["key_embed"][b : b + 1, nodes]
            n = len(nodes)
            x = {
                "inputs": sub_inputs,
                "key_embed": embed,
                "node_embed": embed,
                "key_valid": torch.ones(1, n, dtype=torch.bool),
                "node_valid": torch.ones(1, n, dtype=torch.bool),
                "key_is_node": torch.ones(1, n, dtype=torch.bool),
            }
            x, loop = sec.decoder(x, ["node"])
            loop["final"] = {}
            for task in sec.tasks:
                loop["final"][task.name] = task(x, outputs=loop["final"])
            slots = slice(j * num_queries, (j + 1) * num_queries)
            for (layer, task, key), v in task_outputs(loop).items():
                got = outputs[layer][task][key][b, slots]
                if v.dim() == 3 and v.shape[-1] == n:
                    got = got[:, nodes]
                torch.testing.assert_close(got[None], v, rtol=1e-4, atol=1e-5, msg=str((b, j, layer, task, key)))


def test_out_of_sector_entries_are_filled():
    num_sectors, max_nodes, num_queries = 3, 53, 50
    sec = build(model_config(num_sectors, max_nodes, num_queries, 53))
    inputs = make_inputs([159, 70], num_sectors, max_nodes, seed=4)
    with torch.no_grad():
        outputs = sec(inputs)
    owner = torch.arange(num_sectors).repeat_interleave(num_queries)
    inside = (inputs["node_sector"][:, None, :] == owner[None, :, None]) & inputs["node_valid"][:, None, :]
    for layer in ("layer_1", "final"):
        logit = outputs[layer]["mask"]["pflow_node_logit"]
        assert (logit[~inside] == FINFO_MIN).all()
        assert (logit[inside] > FINFO_MIN).all()
    inc = outputs["final"]["incidence"]["pflow_incidence"]
    assert (inc[~inside] == 0).all()
    # each valid node's incidence is a softmax over the queries of its own sector
    torch.testing.assert_close(inc.sum(1)[inputs["node_valid"]], torch.ones(int(inputs["node_valid"].sum())))


def test_decoder_keeps_sectors_apart():
    num_sectors, max_nodes = 3, 53
    sec = build(model_config(num_sectors, max_nodes, 50, 53))
    inputs = make_inputs([120, 30], num_sectors, max_nodes, seed=5)
    embed = torch.randn(2, 160, 64, requires_grad=True)
    x = {"inputs": inputs, "key_embed": embed, "node_embed": embed, "node_valid": inputs["node_valid"]}
    rows, _ = sec.decode(x)
    for b, j in itertools.product(range(2), range(num_sectors)):
        (grad,) = torch.autograd.grad(rows["query_embed"][b * num_sectors + j].sum(), embed, retain_graph=True)
        own = inputs["node_valid"][b] & (inputs["node_sector"][b] == j)
        node_grad = grad.abs().sum(-1)
        assert (node_grad[b][own] > 0).all()
        assert (node_grad[b][~own] == 0).all()
        assert (node_grad[1 - b] == 0).all()


def test_every_parameter_gets_a_gradient():
    """The configured sector arm: one loss backward reaches every parameter tensor, the encoder through the rows."""
    cfg = yaml.safe_load((CLIC / "studies/sector_decoder/configs/sec3_s42.yaml").read_text())["model"]["model"]
    cfg["init_args"]["matcher"]["init_args"] = {"default_solver": "scipy", "adaptive_solver": False}
    for task in cfg["init_args"]["tasks"]["init_args"]["modules"]:
        if "scale_dict_path" in task["init_args"]:
            task["init_args"]["scale_dict_path"] = str(CLIC / "configs/clic_var_transform.yaml")
    model = build(cfg).train()
    assert isinstance(model, SectoredMaskFormer)
    inputs = make_inputs([150, 90], 3, 53, seed=6)
    targets = make_targets(inputs, 3, 50, [[20, 15, 30], [10, 5, 12]])
    _, _, losses = model.loss(model(inputs), targets)
    sum(v for layer in losses.values() for task in layer.values() for v in task.values()).backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not no_grad
    assert any(n.startswith("encoder.") for n, _ in model.named_parameters())


def test_unknown_task_type_is_refused():
    cfg = model_config(3, 53, 50, 53)
    cfg["init_args"]["tasks"]["init_args"]["modules"].append({
        "class_path": "hepattn.models.task.HitFilterTask",
        "init_args": {"name": "filter", "input_object": "node", "target_field": "valid", "dim": 64},
    })
    with pytest.raises(ValueError, match="output axes of HitFilterTask"):
        build(cfg)


# ---------------------------------------------------------------------------------------------------------------------
# Step 5: block matching


def brute_force(costs, valid_counts, num_queries):
    """Cheapest assignment of each block's valid targets to that block's predictions, by enumeration."""
    total = 0.0
    for j, n in enumerate(valid_counts):
        block = costs[j * num_queries : (j + 1) * num_queries, j * num_queries : (j + 1) * num_queries]
        total += min(sum(block[p[t], t] for t in range(n)) for p in itertools.permutations(range(num_queries)))
    return total


def test_block_match_is_optimal_within_blocks():
    num_blocks, num_queries = 2, 3
    rng = np.random.default_rng(0)
    matcher = Matcher(default_solver="scipy", adaptive_solver=False)
    costs = torch.from_numpy(rng.uniform(0, 1, (40, 6, 6))).float()
    counts = rng.integers(0, num_queries + 1, (40, num_blocks))
    valid = torch.zeros(40, 6, dtype=torch.bool)
    for i, (n0, n1) in enumerate(counts):
        valid[i, :n0] = True
        valid[i, 3 : 3 + n1] = True
    pred = block_match(matcher, costs, valid, num_blocks)
    for i in range(40):
        assert sorted(pred[i].tolist()) == list(range(6))
        assert (pred[i] // num_queries == torch.arange(6) // num_queries).all()
        got = sum(costs[i, pred[i, t], t].item() for t in range(6) if valid[i, t])
        assert got == pytest.approx(brute_force(costs[i].numpy(), counts[i], num_queries), abs=1e-5)
        # the same total as the full assignment with cross-block pairs forbidden
        c = costs[i].numpy().copy()
        c[:3, 3:] = c[3:, :3] = 1e6
        cols = valid[i].numpy()
        rows, assigned = linear_sum_assignment(c[:, cols].T)
        assert got == pytest.approx(c[:, cols].T[rows, assigned].sum(), abs=1e-5)


def test_block_match_ignores_cheaper_cross_block_pairs():
    sec = build(model_config(2, 53, 3, 53))
    costs = torch.ones(1, 6, 6)
    costs[0, 3, 0] = costs[0, 0, 3] = 0.0  # prediction 3 is cheapest for target 0 and prediction 0 for target 3
    valid = torch.tensor([[True, False, False, True, False, False]])
    pred = sec.solve_matching(costs, valid, None)
    assert pred[0, 0] < 3
    assert pred[0, 3] >= 3


# ---------------------------------------------------------------------------------------------------------------------
# Steps 5-6: losses


def flat_losses(losses):
    return {(layer, task, name): v for layer, lo in losses.items() for task, to in lo.items() for name, v in to.items()}


@pytest.mark.parametrize("loss_norm", ["sector_mean", "event"])
def test_one_sector_loss_equals_maskformer(loss_norm):
    ref, sec = pair(1, 160, 150, 160, loss_norm)
    inputs = make_inputs([159, 40, 97], 1, 160, seed=7)
    targets = make_targets(inputs, 1, 150, [[60], [20], [45]], seed=7)
    with torch.no_grad():
        _, _, loss_ref = ref.loss(ref(inputs), targets)
        _, _, loss_sec = sec.loss(sec(inputs), targets)
    a, b = flat_losses(loss_ref), flat_losses(loss_sec)
    assert a.keys() == b.keys()
    assert len(a) >= 10
    for key in a:
        torch.testing.assert_close(b[key], a[key], rtol=1e-5, atol=1e-6, msg=str(key))


def clone_outputs(outputs):
    return {layer: {task: (dict(to) if isinstance(to, dict) else to) for task, to in lo.items()} for layer, lo in outputs.items()}


def hungarian(cost, valid):
    """(pred, true) costs -> prediction index per target slot: Hungarian on the valid prefix, unused predictions after."""
    n_valid = int(valid.sum())
    assert valid[:n_valid].all()
    _, pred = linear_sum_assignment(cost[:, :n_valid].T.double().numpy())
    rest = [p for p in range(cost.shape[0]) if p not in set(pred.tolist())]
    return torch.tensor(pred.tolist() + rest)


def permute(layer_outputs, pred):
    batch = pred.shape[0]
    for task, to in layer_outputs.items():
        if task != "attn_mask":
            layer_outputs[task] = {k: v[torch.arange(batch)[:, None], pred] for k, v in to.items()}


def per_sector_loop_loss(model, outputs, targets, num_sectors, num_queries):
    """Independent loop: each sector's rows built node by node, then reference costs, Hungarian and losses; 1 / S mean."""
    batch = targets["node_valid"].shape[0]
    max_nodes = model.max_nodes_per_sector
    total = {}
    for j in range(num_sectors):
        slots = slice(j * num_queries, (j + 1) * num_queries)
        node_lists = []
        for b in range(batch):
            members = torch.nonzero(targets["node_valid"][b] & (targets["node_sector"][b] == j)).flatten()
            node_lists.append(members[torch.argsort(targets["node_sector_rank"][b, members])])

        def rows(t, fill, slots=slots, node_lists=node_lists):
            out = torch.full((batch, num_queries, max_nodes), fill, dtype=t.dtype)
            for b, nodes in enumerate(node_lists):
                out[b, :, : len(nodes)] = t[b, slots][:, nodes]
            return out

        fills = {"pflow_node_logit": FINFO_MIN, "pflow_incidence": 0.0}
        sub_out = {
            layer: {
                task: {k: rows(v, fills[k]) if k in fills else v[:, slots] for k, v in to.items()} for task, to in lo.items() if task != "attn_mask"
            }
            for layer, lo in outputs.items()
            if layer != "encoder"
        }
        sub_t = {"node_valid": torch.zeros(batch, max_nodes, dtype=torch.bool)}
        for b, nodes in enumerate(node_lists):
            sub_t["node_valid"][b, : len(nodes)] = True
        for key, v in targets.items():
            if key.startswith("particle_"):
                sub_t[key] = rows(v, 0) if v.dim() == 3 else v[:, slots]
        for layer, cost in MaskFormer._compute_decoder_costs(model, sub_out, sub_t).items():  # noqa: SLF001
            permute(sub_out[layer], torch.stack([hungarian(cost[b], sub_t["particle_valid"][b]) for b in range(batch)]))
        has_targets = bool(sub_t["particle_valid"].any())
        for key, v in flat_losses(MaskFormer._compute_decoder_losses(model, sub_out, sub_t)).items():  # noqa: SLF001
            total[key] = total.get(key, 0.0) + (v if has_targets else torch.where(v.isnan(), 0.0, v)) / num_sectors
    return total


def event_loop_loss(model, outputs, targets, num_sectors, num_queries):
    """Reference costs and losses on the event-frame outputs, Hungarian per sector block."""
    batch = targets["node_valid"].shape[0]
    decoder_outputs = {k: v for k, v in outputs.items() if k != "encoder"}
    for layer, cost in MaskFormer._compute_decoder_costs(model, decoder_outputs, targets).items():  # noqa: SLF001
        pred = torch.zeros(batch, num_sectors * num_queries, dtype=torch.long)
        for b, j in itertools.product(range(batch), range(num_sectors)):
            s = slice(j * num_queries, (j + 1) * num_queries)
            pred[b, s] = j * num_queries + hungarian(cost[b, s, s], targets["particle_valid"][b, s])
        permute(decoder_outputs[layer], pred)
    return flat_losses(MaskFormer._compute_decoder_losses(model, decoder_outputs, targets))  # noqa: SLF001


def sector_case(loss_norm, targets_per_sector, num_nodes=(150, 90), seed=8):
    ref, sec = pair(3, 53, 50, 53, loss_norm)
    del ref
    inputs = make_inputs(list(num_nodes), 3, 53, seed=seed)
    targets = make_targets(inputs, 3, 50, targets_per_sector, seed=seed)
    with torch.no_grad():
        outputs = sec(inputs)
    return sec, inputs, outputs, targets


def assert_losses_close(got, expected):
    assert got.keys() == expected.keys()
    assert len(got) >= 10
    for key in got:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-5, atol=1e-6, msg=str(key))


def test_sector_mean_equals_a_per_sector_loop():
    sec, _, outputs, targets = sector_case("sector_mean", [[20, 15, 30], [10, 5, 12]])
    expected = per_sector_loop_loss(sec, clone_outputs(outputs), targets, 3, 50)
    _, _, losses = sec.loss(clone_outputs(outputs), targets)
    assert_losses_close(flat_losses(losses), expected)


def test_event_mode_equals_reference_losses_on_event_outputs():
    sec, _, outputs, targets = sector_case("event", [[20, 15, 30], [10, 5, 12]])
    expected = event_loop_loss(sec, clone_outputs(outputs), targets, 3, 50)
    _, _, losses = sec.loss(clone_outputs(outputs), targets)
    assert_losses_close(flat_losses(losses), expected)


def test_loss_norm_option_changes_the_normalisation():
    """Same weights and inputs: the per-sector mask BCE and incidence KL normalise by the sector's nodes, the event
    ones by the event's nodes (out-of-sector entries add 0), so the sector mean is about S times larger.
    """
    sec_mean, _, outputs, targets = sector_case("sector_mean", [[20, 15, 30], [10, 5, 12]])
    sec_event, _, outputs_event, _ = sector_case("event", [[20, 15, 30], [10, 5, 12]])
    torch.testing.assert_close(outputs_event["final"]["mask"]["pflow_node_logit"], outputs["final"]["mask"]["pflow_node_logit"])
    _, _, mean = sec_mean.loss(clone_outputs(outputs), targets)
    _, _, event = sec_event.loss(clone_outputs(outputs), targets)
    assert mean["final"]["mask"]["mask_bce"] > 1.5 * event["final"]["mask"]["mask_bce"]
    assert mean["final"]["incidence"]["kl_div"] > 1.5 * event["final"]["incidence"]["kl_div"]


def test_sector_without_targets():
    """Sector 2 has nodes but no targets in either event: it adds its no-object classification loss, 0 for the
    terms over valid targets, and still counts in the 1 / S mean.
    """
    sec, _, outputs, targets = sector_case("sector_mean", [[20, 15, 0], [10, 5, 0]])
    assert targets["node_valid"].logical_and(targets["node_sector"] == 2).any(-1).all()
    expected = per_sector_loop_loss(sec, clone_outputs(outputs), targets, 3, 50)
    sec.train()
    inputs = make_inputs([150, 90], 3, 53, seed=8)
    _, _, losses = sec.loss(sec(inputs), targets)
    flat = flat_losses(losses)
    assert all(torch.isfinite(v) for v in flat.values())
    assert_losses_close({k: v.detach() for k, v in flat.items()}, expected)
    sum(flat.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in sec.parameters() if p.grad is not None)


def test_row_without_nodes():
    """An event with fewer nodes than sectors gives a decoder row with no node: outputs and losses stay finite."""
    sec, inputs, outputs, targets = sector_case("sector_mean", [[20, 15, 30], [1, 1, 0]], num_nodes=(150, 2))
    assert not (inputs["node_valid"][1] & (inputs["node_sector"][1] == 2)).any()
    for v in task_outputs(outputs).values():
        assert not v.isnan().any()
    _, _, losses = sec.loss(clone_outputs(outputs), targets)
    assert all(torch.isfinite(v) for v in flat_losses(losses).values())
