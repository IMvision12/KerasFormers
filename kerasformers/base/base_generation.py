import keras
from keras import ops

from kerasformers.samplers import GreedySampler


class BaseGeneration:
    """Backend-agnostic autoregressive generation for decoder-only LMs (mirrors HF's ``GenerationMixin``).

    A mixin added to a subclassed decoder backbone (e.g. :class:`Qwen3Model`, and
    future Granite) to give it a fast ``generate``. It bundles the shared, optimized
    cross-backend decode engine with the decoder-only entry points (the prompt is the
    input token ids). :class:`Seq2SeqGeneration` subclasses this and overrides
    ``generate`` / ``generate_step`` for encoder-decoder models (Whisper, Speech2Text),
    reusing the same engine.

    A model plugs in two hooks:

    - ``build_cache(token_ids, padding_mask, max_len) -> (cache, logits)`` -- the
      parallel prefill: populate a pre-allocated fixed-size KV cache (any opaque tensor
      the model defines) and return it plus the last-token logits.
    - ``call_with_cache(token_ids, cache, cache_update_index) -> (logits, cache)`` --
      one decode step that reads/writes the cache at the given index.

    Performance comes from a single fused decode loop (``keras.ops.while_loop`` over a
    constant-shape cache) wrapped in a per-backend compiled function -- ``jax.jit`` with
    stateless variable threading on JAX, ``tf.function(jit_compile=True)`` on
    TensorFlow, eager on Torch -- cached on the instance. Decoding strategy is a
    pluggable :class:`~kerasformers.samplers.Sampler` (greedy by default); for
    stochastic samplers the random noise is drawn once *outside* the loop (via a
    ``SeedGenerator``) and consumed with the Gumbel-max trick, so generation stays
    identical across backends. Output is a fixed ``(batch, max_new_tokens)`` padded with
    the eos id after a sequence finishes. A model may set the ``eos_token_id`` class
    attr for its default stop token(s); explicit ``generate`` arguments win over it.
    """

    # Per-model decoding default: stop token id(s). Override as a class attr (e.g.
    # Qwen3CausalLM sets eos_token_id = (151645,)). max_new_tokens / seed default
    # library-wide (128 / 0) and are normally passed per call.
    eos_token_id = ()

    def build_cache(self, token_ids, padding_mask, max_len):
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_cache()."
        )

    def call_with_cache(self, token_ids, cache, cache_update_index):
        raise NotImplementedError(
            f"{type(self).__name__} must implement call_with_cache()."
        )

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=None,
        eos_token_id=None,
        sampler=None,
        seed=None,
    ):
        max_new_tokens, eos, sampler, seed = self.resolve_generation_args(
            max_new_tokens, eos_token_id, sampler, seed
        )
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch = int(input_ids.shape[0])
        padding_mask = (
            None
            if attention_mask is None
            else ops.cast(ops.convert_to_tensor(attention_mask), "int32")
        )
        cache_key = (max_new_tokens, eos, attention_mask is not None, id(sampler))
        fn = self.cached_generate_function(cache_key, max_new_tokens, eos, sampler)
        noise = self.draw_noise(sampler, max_new_tokens, batch, seed)
        return self.run_compiled(fn, (input_ids, padding_mask), noise)

    def generate_step(
        self, token_ids, padding_mask, noise, max_new_tokens, eos, sampler
    ):
        token_ids = ops.cast(ops.convert_to_tensor(token_ids), "int32")
        prompt_len = int(token_ids.shape[1])
        cache, logits = self.build_cache(
            token_ids, padding_mask, prompt_len + max_new_tokens
        )
        return self.decode_loop(
            cache, logits, prompt_len, noise, max_new_tokens, eos, sampler
        )

    def decode_loop(
        self, cache, logits, prompt_len, noise, max_new_tokens, eos, sampler
    ):
        # Shared inner loop: given the prefill's (cache, last-token logits), sample the
        # first token then run a fixed-shape ``while_loop`` of single-token decode steps.
        batch = int(logits.shape[0])
        first_tok = ops.cast(
            sampler.sample(logits, ops.take(noise, 0, axis=0)), "int32"
        )[:, None]
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
            step_noise = ops.take(noise, i + 1, axis=0)
            nxt = ops.cast(sampler.sample(logits, step_noise), "int32")[:, None]
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

    def make_generate_function(self, max_new_tokens, eos, sampler):
        # Wrap generate_step in a per-backend compiled callable run(runtime_args, noise).
        # runtime_args is the flavor's tuple of input tensors (the only things that
        # vary at run time); max_new_tokens/eos/sampler are baked in (static).
        backend = keras.backend.backend()
        if backend == "jax":
            import itertools

            import jax

            def compiled(runtime_args, noise, state):
                trainable, non_trainable = state
                mapping = itertools.chain(
                    zip(self.trainable_variables, trainable),
                    zip(self.non_trainable_variables, non_trainable),
                )
                # Variables threaded as donated args (sharding-safe, no
                # constant-baking) rather than closed over -- KerasHub's pattern.
                with keras.StatelessScope(state_mapping=mapping):
                    return self.generate_step(
                        *runtime_args, noise, max_new_tokens, eos, sampler
                    )

            compiled = jax.jit(compiled)

            def run(runtime_args, noise):
                state = (
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                return compiled(runtime_args, noise, state)

            return run

        if backend == "tensorflow":
            import tensorflow as tf

            return tf.function(
                lambda runtime_args, noise: self.generate_step(
                    *runtime_args, noise, max_new_tokens, eos, sampler
                ),
                jit_compile=True,
            )

        def run(runtime_args, noise):
            return self.generate_step(
                *runtime_args, noise, max_new_tokens, eos, sampler
            )

        return run

    def resolve_generation_args(self, max_new_tokens, eos_token_id, sampler, seed):
        # Merge explicit args over per-model / library defaults, and normalize eos to
        # a tuple of ints.
        if max_new_tokens is None:
            max_new_tokens = 128
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        if sampler is None:
            sampler = GreedySampler()
        if seed is None:
            seed = 0
        eos = tuple(
            int(e)
            for e in (
                eos_token_id
                if isinstance(eos_token_id, (list, tuple))
                else [eos_token_id]
            )
        )
        return int(max_new_tokens), eos, sampler, int(seed)

    def draw_noise(self, sampler, max_new_tokens, batch, seed):
        # Stochastic samplers consume per-step uniform noise drawn once here (a
        # SeedGenerator works on every backend); greedy gets a trivial placeholder.
        if sampler.stochastic:
            return keras.random.uniform(
                (max_new_tokens, batch, int(self.vocab_size)),
                seed=keras.random.SeedGenerator(int(seed)),
            )
        return ops.zeros((max_new_tokens, batch, 1), dtype="float32")

    def cached_generate_function(self, cache_key, max_new_tokens, eos, sampler):
        # Compiled fn cached per (max_new_tokens, eos, has_mask, sampler); `noise` is a
        # runtime arg so re-seeding never recompiles.
        fns = self.__dict__.setdefault("_generate_functions", {})
        fn = fns.get(cache_key)
        if fn is None:
            fn = self.make_generate_function(max_new_tokens, eos, sampler)
            fns[cache_key] = fn
        return fn

    def run_compiled(self, fn, runtime_args, noise):
        if keras.backend.backend() == "torch":
            import torch

            with torch.no_grad():
                out = fn(runtime_args, noise)
        else:
            out = fn(runtime_args, noise)
        return ops.convert_to_numpy(out)
