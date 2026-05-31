from dataclasses import dataclass, field

from kerasformers.base.samplers import GreedySampler, Sampler


@dataclass
class GenerationConfig:
    """Decoding defaults for a causal LM, separate from the model's weights/config.

    A model may set a class- or instance-level ``generation_config`` (e.g. its eos
    id); :meth:`CausalLM.generate` falls back to it when the corresponding argument
    is not passed explicitly. Keeps model-specific decoding constants out of the
    shared :class:`CausalLM` base.
    """

    max_new_tokens: int = 128
    eos_token_id: tuple = ()
    sampler: Sampler = field(default_factory=GreedySampler)
    seed: int = 0
