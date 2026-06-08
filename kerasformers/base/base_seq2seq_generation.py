from keras import ops

from kerasformers.base.base_generation import BaseGeneration


class Seq2SeqGeneration(BaseGeneration):
    """Encoder-decoder flavor of :class:`BaseGeneration` (Whisper / Speech2Text / ...).

    Same optimized cross-backend decode engine as :class:`BaseGeneration`, but the prefill
    first runs an encoder over the source (audio features, source tokens, ...) and the
    decoder cross-attends to that frozen context at every step. The decoder "prompt"
    is the start / forced tokens rather than a user prompt. A model implements three
    hooks:

    - ``encode(encoder_inputs) -> encoder_hidden_states`` -- run the encoder once.
    - ``build_cache(decoder_start_ids, encoder_hidden_states, max_len) -> (cache, logits)``
      -- build the static cross-attention KV from ``encoder_hidden_states`` and prefill
      the decoder start tokens into a fixed self-attention KV cache (both carried inside
      the opaque ``cache``); return it plus the last-token logits.
    - ``call_with_cache(token_ids, cache, cache_update_index) -> (logits, cache)`` --
      one cached decode step (self-attn reads/writes the cache; cross-attn reads the
      static cross-KV already inside ``cache``).

    NOTE: this is the shared contract; no model is wired onto it yet. Whisper and
    Speech2Text still use their own ``generate()`` until their functional decoders gain
    cache-capable attention. Constraints like Whisper's forced-decoder-ids and
    suppress-tokens will be folded into the decode path when those models are migrated.
    """

    def greedy_decode(
        self,
        encoder_hidden_states,
        decoder_start_token_id,
        eos_token_id,
        max_new_tokens,
        forced_ids=None,
        logits_processor=None,
    ):
        """Shared **cacheless** greedy seq2seq decode for the functional ASR
        decoders (Whisper / Speech2Text / Moonshine) that don't yet have a KV
        cache. ``self.decoder`` is called with ``{"decoder_input_ids",
        "encoder_hidden_states"}`` over the full growing prefix each step.

        ``forced_ids`` maps a decoder position -> a forced token id (Whisper
        prompt ids); at those positions the decoder is skipped. ``logits_processor``
        is ``(step, next_logits) -> next_logits`` for biasing the last-token logits
        (Whisper suppress / begin-suppress). Greedy (argmax); a row that emits
        ``eos`` is padded with ``eos`` thereafter. Returns the ``(batch, len)`` id
        tensor including the start token.

        This is the shared seam; the compiled fixed-cache engine
        (``build_cache`` / ``call_with_cache``) remains the target once these
        decoders gain cache-capable attention.
        """
        forced_ids = forced_ids or {}
        batch = encoder_hidden_states.shape[0]
        generated = ops.full((batch, 1), decoder_start_token_id, dtype="int32")
        done = ops.zeros((batch,), dtype="bool")
        for step in range(max_new_tokens):
            cur_pos = generated.shape[1]
            if cur_pos in forced_ids:
                next_ids = ops.full((batch,), forced_ids[cur_pos], dtype="int32")
            else:
                logits = self.decoder(
                    {
                        "decoder_input_ids": generated,
                        "encoder_hidden_states": encoder_hidden_states,
                    }
                )
                next_logits = logits[:, -1, :]
                if logits_processor is not None:
                    next_logits = logits_processor(step, next_logits)
                next_ids = ops.cast(ops.argmax(next_logits, axis=-1), "int32")
            next_ids = ops.cast(ops.where(done, eos_token_id, next_ids), "int32")
            generated = ops.concatenate([generated, next_ids[:, None]], axis=1)
            done = ops.logical_or(done, ops.equal(next_ids, eos_token_id))
            if bool(ops.all(done)):
                break
        return generated

    def encode(self, encoder_inputs):
        raise NotImplementedError(f"{type(self).__name__} must implement encode().")

    def generate_step(
        self, encoder_inputs, decoder_start_ids, noise, max_new_tokens, eos, sampler
    ):
        decoder_start_ids = ops.cast(ops.convert_to_tensor(decoder_start_ids), "int32")
        prompt_len = int(decoder_start_ids.shape[1])
        encoder_hidden_states = self.encode(encoder_inputs)
        cache, logits = self.build_cache(
            decoder_start_ids, encoder_hidden_states, prompt_len + max_new_tokens
        )
        return self.decode_loop(
            cache, logits, prompt_len, noise, max_new_tokens, eos, sampler
        )

    def generate(
        self,
        encoder_inputs,
        decoder_input_ids,
        max_new_tokens=None,
        eos_token_id=None,
        sampler=None,
        seed=None,
    ):
        max_new_tokens, eos, sampler, seed = self.resolve_generation_args(
            max_new_tokens, eos_token_id, sampler, seed
        )
        decoder_input_ids = ops.cast(ops.convert_to_tensor(decoder_input_ids), "int32")
        batch = int(decoder_input_ids.shape[0])
        cache_key = (max_new_tokens, eos, id(sampler))
        fn = self.cached_generate_function(cache_key, max_new_tokens, eos, sampler)
        noise = self.draw_noise(sampler, max_new_tokens, batch, seed)
        return self.run_compiled(fn, (encoder_inputs, decoder_input_ids), noise)
