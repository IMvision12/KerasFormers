from keras import ops

from kerasformers.base.generation import BaseGeneration


class CausalLM(BaseGeneration):
    """Decoder-only flavor of :class:`BaseGeneration` (LLMs: Qwen, Granite, ...).

    A mixin added to a subclassed decoder backbone (e.g. :class:`Qwen3Model`) to give
    it a fast ``generate``. The prompt is the input token ids; the model supplies two
    hooks (see :class:`BaseGeneration` for the full contract):

    - ``build_cache(token_ids, padding_mask, max_len) -> (cache, logits)`` -- parallel
      prefill of the prompt into a pre-allocated fixed-size KV cache.
    - ``call_with_cache(token_ids, cache, cache_update_index) -> (logits, cache)`` --
      one cached decode step.

    All the optimized cross-backend machinery (compiled fused decode loop, sampler +
    pre-drawn noise, fixed padded output) is inherited from :class:`BaseGeneration`.
    """

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
