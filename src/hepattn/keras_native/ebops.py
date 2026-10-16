"""EBOPs read-out, beta writes and the EBOPs on/off switch for the stateless JAX train step (plan section 3.6).

Why the read-out is not ``_ebops``: HGQ2 stores each leaf's EBOPs in a ``uint32`` variable
(``hgq/layers/core/base.py``: ``add_weight(name='ebops', dtype='uint32')``, written with ``ops.cast``). On JAX the cast
saturates at 4294967295 (``UINT32_MAX``); the Keras torch backend stores ``uint32`` as ``int64``, so the hybrid reads exact
values. The float value enters the loss as ``ebops * beta`` (``add_loss``), so the exact read-out is::

    EBOPs = penalty / beta,   penalty = sum of the EBOPs terms of ``stateless_call(..., return_losses=True)``

which holds when every leaf has the same beta (``set_beta`` writes one value; ``read_beta`` checks it). It is valid only on
EBOPs-active training steps and needs beta > 0.

The loss list of ``stateless_call`` is ``Layer.losses``: the ``add_loss`` terms of the model and its sublayers first, then
one term per trainable weight that has a regularizer (``keras/src/layers/layer.py`` ``losses`` and
``_get_regularization_losses``). HGQ2 gives some quantizer variables regularizers; these terms do not depend on beta
(0.335 of 804.91 at one test spec, ``test_ebops.py``; the plan's 0.381 / 804.96 include the hybrid's double count), so
``split_losses`` drops them before the division. The native model calls no other ``add_loss``, so every term in front is
an EBOPs term (``tests/keras_native/test_ebops.py`` checks both facts).

``EbopsLeaves.enabled(on)`` sets ``_enable_ebops`` on every leaf while a step is traced. HGQ2 reads that Python bool at
trace time, so a step jitted with a static ``ebops_on`` argument has one executable per value (PeriodicEBOPs).
"""

import contextlib
from collections.abc import Iterator, Sequence

import jax.numpy as jnp
import numpy as np

UINT32_MAX = 2**32 - 1


class EbopsLeaves:
    """The HGQ2 leaves of a built model that carry an EBOPs term (a ``_beta`` variable), with their state positions.

    Args:
        model: A built Keras model (all HGQ2 layers created, e.g. by one eager call).

    Raises:
        ValueError: If a leaf's beta or ebops variable is not among the model's non-trainable variables.
    """

    def __init__(self, model):
        leaves, seen = [], set()
        for layer in model._flatten_layers(include_self=False):  # noqa: SLF001
            if getattr(layer, "_beta", None) is not None and id(layer) not in seen:
                seen.add(id(layer))
                leaves.append(layer)
        self.leaves = leaves
        position = {id(v): i for i, v in enumerate(model.non_trainable_variables)}
        missing = [leaf.name for leaf in leaves if id(leaf._beta) not in position or id(leaf._ebops) not in position]  # noqa: SLF001
        if missing:
            raise ValueError(f"beta/ebops variables of {missing[:5]} are not non-trainable variables of the model")
        self.beta_index = [position[id(leaf._beta)] for leaf in leaves]  # noqa: SLF001
        self.ebops_index = [position[id(leaf._ebops)] for leaf in leaves]  # noqa: SLF001
        self.n_regularizers = sum(1 for v in model.trainable_weights if getattr(v, "regularizer", None) is not None)

    def __len__(self) -> int:
        return len(self.leaves)

    @contextlib.contextmanager
    def enabled(self, on: bool) -> Iterator[None]:
        """Set every leaf's ``_enable_ebops`` to ``on`` for the duration (a trace), then restore the previous flags."""
        before = [leaf._enable_ebops for leaf in self.leaves]  # noqa: SLF001
        try:
            for leaf in self.leaves:
                leaf._enable_ebops = bool(on)  # noqa: SLF001
            yield
        finally:
            for leaf, flag in zip(self.leaves, before, strict=True):
                leaf._enable_ebops = flag  # noqa: SLF001

    def split_losses(self, losses: Sequence) -> tuple[list, list]:
        """(EBOPs terms, regularizer terms) of a ``stateless_call`` loss list; the regularizer terms are the last ones.

        Raises:
            ValueError: If the list is shorter than the number of regularized weights.
        """
        losses = list(losses)
        n_pen = len(losses) - self.n_regularizers
        if n_pen < 0:
            raise ValueError(f"{len(losses)} loss terms but {self.n_regularizers} regularized weights")
        return losses[:n_pen], losses[n_pen:]

    def set_beta(self, ntv: Sequence, beta: float) -> list:
        """A copy of ``ntv`` with every leaf's beta set to ``beta`` (same shape and dtype: the jitted step does not retrace)."""
        ntv = list(ntv)
        for i in self.beta_index:
            ntv[i] = jnp.full(np.shape(ntv[i]), beta, dtype=ntv[i].dtype)
        return ntv

    def read_beta(self, ntv: Sequence) -> float:
        """The common beta of every leaf.

        Raises:
            ValueError: If the leaves hold different betas (then ``penalty / beta`` is not the EBOPs).
        """
        values = np.unique(np.asarray([np.asarray(ntv[i], dtype=np.float64) for i in self.beta_index]))
        if values.size != 1:
            raise ValueError(f"the EBOPs leaves hold {values.size} different betas (min {values.min()}, max {values.max()})")
        return float(values[0])

    def stored_ebops(self, ntv: Sequence) -> np.ndarray:
        """Each leaf's stored ``_ebops`` (uint32: saturated at ``UINT32_MAX`` on JAX). For the saturation control only."""
        return np.asarray([np.asarray(ntv[i]) for i in self.ebops_index], dtype=np.uint64)


def ebops_readout(penalty: float, beta: float) -> float:
    """Total EBOPs from the summed EBOPs terms of an EBOPs-active step and the common beta.

    Raises:
        ValueError: If ``beta <= 0`` (no read-out exists; use ``min_beta > 0`` and a ramp that starts above 0).
    """
    if not beta > 0:
        raise ValueError(f"EBOPs read-out needs beta > 0, got {beta}")
    return float(penalty) / float(beta)
