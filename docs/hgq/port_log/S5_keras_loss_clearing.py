"""Keras loss clearing in ``stateless_call``: a sublayer called twice from the model's ``call`` keeps one ``add_loss`` term
(the last call's: each call is the entry layer of its own call context, which clears its earlier terms); called twice
inside one other layer it keeps both.

    KERAS_BACKEND=jax <hepattn-keras-jax>/.venv-jax/bin/python docs/hgq/port_log/S5_keras_loss_clearing.py

Output (S5 fixer 2, keras 3.15.0 in .venv-jax): ``Direct 1 [4.0]`` and ``Nested 2 [2.0, 4.0]``.
"""

import keras
from keras import ops


class AddsLoss(keras.layers.Layer):
    def call(self, x):
        self.add_loss(ops.sum(x))
        return x + 1.0


class Wrap(keras.layers.Layer):
    def __init__(self, inner, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def call(self, x):
        return self.inner(self.inner(x))


class Direct(keras.Model):
    def __init__(self):
        super().__init__()
        self.inner = AddsLoss()

    def call(self, x):
        return self.inner(self.inner(x))


class Nested(keras.Model):
    def __init__(self):
        super().__init__()
        self.wrap = Wrap(AddsLoss())

    def call(self, x):
        return self.wrap(x)


x = ops.ones((1, 2))
for model_cls in (Direct, Nested):
    model = model_cls()
    model(x)
    _, _, losses = model.stateless_call([], [v.value for v in model.non_trainable_variables], x, return_losses=True)
    print(model_cls.__name__, len(losses), [float(t) for t in losses])
