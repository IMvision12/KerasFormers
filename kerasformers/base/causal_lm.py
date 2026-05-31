import keras
from keras import ops

_MASK_NEG = -1e9


class CausalLM:
    """Greedy autoregressive generation for subclassed causal language models.

    A mixin added to a subclassed backbone (e.g. :class:`Qwen3Model`) to give it a
    fast, backend-agnostic ``generate``. The model supplies two hooks:

    - ``build_cache(token_ids, padding_mask, max_len) -> (cache, logits)`` -- the
      parallel prefill: populate a pre-allocated fixed-size cache (any opaque tensor
      the model defines) and return it plus the last-token logits.
    - ``call_with_cache(token_ids, cache, cache_update_index) -> (logits, cache)``
      -- one decode step that reads/writes the cache at the given index.

    The shared machinery mirrors KerasHub's ``CausalLM``: a single fused decode loop
    (``keras.ops.while_loop`` over a constant-shape cache) wrapped in a per-backend
    compiled function -- ``jax.jit`` with stateless variable threading on JAX,
    ``tf.function(jit_compile=True)`` on TensorFlow, eager on Torch -- cached on the
    instance. Greedy (argmax). Output is a fixed ``(batch, max_new_tokens)`` padded
    with the eos id after a sequence finishes.
    """

    def build_cache(self, token_ids, padding_mask, max_len):
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_cache()."
        )

    def call_with_cache(self, token_ids, cache, cache_update_index):
        raise NotImplementedError(
            f"{type(self).__name__} must implement call_with_cache()."
        )

    def generate_step(self, token_ids, padding_mask, max_new_tokens, eos):
        token_ids = ops.cast(ops.convert_to_tensor(token_ids), "int32")
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        max_len = prompt_len + max_new_tokens
        cache, logits = self.build_cache(token_ids, padding_mask, max_len)
        first_tok = ops.cast(ops.argmax(logits, axis=-1), "int32")[:, None]
        first_eos = eos[0] if eos else 0
        if max_new_tokens <= 1:
            return first_tok

        done = ops.zeros((batch,), dtype="bool")
        for e in eos:
            done = ops.logical_or(done, first_tok[:, 0] == e)
        steps = max_new_tokens - 1
        # Tokens 1..steps; pre-filled with the eos id so an early stop leaves eos
        # padding and the output stays a fixed (batch, max_new_tokens).
        buf = ops.full((steps, batch, 1), first_eos, dtype="int32")

        def cond(i, tok, cache, pos, done, buf):
            return ops.logical_and(i < steps, ops.logical_not(ops.all(done)))

        def body(i, tok, cache, pos, done, buf):
            logits, cache = self.call_with_cache(tok, cache, pos)
            nxt = ops.cast(ops.argmax(logits, axis=-1), "int32")[:, None]
            nxt = ops.cast(ops.where(done[:, None], first_eos, nxt), "int32")
            for e in eos:
                done = ops.logical_or(done, nxt[:, 0] == e)
            buf = ops.slice_update(buf, (i, 0, 0), nxt[None])
            return (i + 1, nxt, cache, pos + 1, done, buf)

        init = (
            ops.convert_to_tensor(0, dtype="int32"),
            first_tok,
            cache,
            ops.convert_to_tensor(prompt_len, dtype="int32"),
            done,
            buf,
        )
        buf = ops.while_loop(cond, body, init, maximum_iterations=steps)[-1]
        tail = ops.transpose(buf[:, :, 0], (1, 0))  # (batch, steps)
        return ops.concatenate([first_tok, tail], axis=1)

    def make_generate_function(self, max_new_tokens, eos):
        backend = keras.backend.backend()
        if backend == "jax":
            import itertools

            import jax

            def compiled(token_ids, padding_mask, state):
                trainable, non_trainable = state
                mapping = itertools.chain(
                    zip(self.trainable_variables, trainable),
                    zip(self.non_trainable_variables, non_trainable),
                )
                # Variables are threaded in as donated args (sharding-safe, no
                # constant-baking) rather than closed over -- KerasHub's pattern.
                with keras.StatelessScope(state_mapping=mapping):
                    return self.generate_step(
                        token_ids, padding_mask, max_new_tokens, eos
                    )

            compiled = jax.jit(compiled)

            def run(token_ids, padding_mask):
                state = (
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                return compiled(token_ids, padding_mask, state)

            return run

        if backend == "tensorflow":
            import tensorflow as tf

            return tf.function(
                lambda token_ids, padding_mask: self.generate_step(
                    token_ids, padding_mask, max_new_tokens, eos
                ),
                jit_compile=True,
            )

        def run(token_ids, padding_mask):
            return self.generate_step(token_ids, padding_mask, max_new_tokens, eos)

        return run

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=128, eos_token_id=(151645,)
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        padding_mask = (
            None
            if attention_mask is None
            else ops.cast(ops.convert_to_tensor(attention_mask), "int32")
        )
        eos = tuple(
            int(e)
            for e in (
                eos_token_id
                if isinstance(eos_token_id, (list, tuple))
                else [eos_token_id]
            )
        )
        # Bypass keras attribute-tracking for the plain dict of compiled fns; cache
        # per (max_new_tokens, eos, has_mask) so repeated calls reuse the executable.
        fns = self.__dict__.setdefault("_generate_functions", {})
        key = (int(max_new_tokens), eos, attention_mask is not None)
        fn = fns.get(key)
        if fn is None:
            fn = self.make_generate_function(max_new_tokens, eos)
            fns[key] = fn
        if keras.backend.backend() == "torch":
            import torch

            with torch.no_grad():
                out = fn(input_ids, padding_mask)
        else:
            out = fn(input_ids, padding_mask)
        return ops.convert_to_numpy(out)
