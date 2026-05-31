from keras import ops


class Sampler:
    """Maps logits ``(batch, vocab)`` + per-step uniform ``noise`` to next ids.

    ``stochastic`` tells :class:`CausalLM` whether to pre-compute random noise for
    the whole decode (cross-backend, *outside* the compiled loop, via a single
    ``SeedGenerator``). Greedy ignores the noise; stochastic samplers turn it into
    a draw with the Gumbel-max trick (``argmax(masked_logits + gumbel(noise))``),
    so no RNG runs inside the fused ``while_loop`` and the result is identical on
    TF / JAX / Torch.
    """

    stochastic = False

    def sample(self, logits, noise):
        raise NotImplementedError(f"{type(self).__name__} must implement sample().")

    def get_config(self):
        return {}


def gumbel(noise):
    # uniform(0, 1) -> Gumbel(0, 1)
    u = ops.clip(noise, 1e-9, 1.0)
    return -ops.log(-ops.log(u))


class GreedySampler(Sampler):
    """Deterministic argmax — the default."""

    stochastic = False

    def sample(self, logits, noise):
        return ops.cast(ops.argmax(logits, axis=-1), "int32")


class TopKSampler(Sampler):
    """Sample from the ``k`` highest-probability tokens (temperature-scaled)."""

    stochastic = True

    def __init__(self, k=50, temperature=1.0):
        self.k = int(k)
        self.temperature = float(temperature)

    def sample(self, logits, noise):
        logits = ops.cast(logits, "float32") / self.temperature
        kth = ops.min(ops.top_k(logits, k=self.k)[0], axis=-1, keepdims=True)
        masked = ops.where(logits < kth, ops.full_like(logits, -1e9), logits)
        return ops.cast(ops.argmax(masked + gumbel(noise), axis=-1), "int32")

    def get_config(self):
        return {"k": self.k, "temperature": self.temperature}


class TopPSampler(Sampler):
    """Nucleus sampling — smallest set of tokens whose cumulative prob >= ``p``."""

    stochastic = True

    def __init__(self, p=0.9, temperature=1.0):
        self.p = float(p)
        self.temperature = float(temperature)

    def sample(self, logits, noise):
        logits = ops.cast(logits, "float32") / self.temperature
        order = ops.argsort(-logits, axis=-1)  # descending
        sorted_logits = ops.take_along_axis(logits, order, axis=-1)
        probs = ops.softmax(sorted_logits, axis=-1)
        cumulative = ops.cumsum(probs, axis=-1) - probs  # exclusive prefix
        keep_sorted = cumulative < self.p  # nucleus (sorted)
        inverse = ops.argsort(order, axis=-1)  # scatter back
        keep = ops.take_along_axis(keep_sorted, inverse, axis=-1)  # vocab order
        masked = ops.where(keep, logits, ops.full_like(logits, -1e9))
        return ops.cast(ops.argmax(masked + gumbel(noise), axis=-1), "int32")

    def get_config(self):
        return {"p": self.p, "temperature": self.temperature}
