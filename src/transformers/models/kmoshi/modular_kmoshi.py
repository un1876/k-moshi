"""PyTorch Kmoshi model."""

from ..personaplex.configuration_personaplex import PersonaplexConfig, PersonaplexDepthConfig
from ..personaplex.modeling_personaplex import (
    PersonaplexForCausalLM,
    PersonaplexForConditionalGeneration,
    PersonaplexModel,
)


class KmoshiDepthConfig(PersonaplexDepthConfig):
    pass


class KmoshiConfig(PersonaplexConfig):
    pass


class KmoshiModel(PersonaplexModel):
    pass


class KmoshiForCausalLM(PersonaplexForCausalLM):
    pass


class KmoshiForConditionalGeneration(PersonaplexForConditionalGeneration):
    pass


__all__ = [
    "KmoshiConfig",
    "KmoshiDepthConfig",
    "KmoshiModel",
    "KmoshiForCausalLM",
    "KmoshiForConditionalGeneration",
    "KmoshiPreTrainedModel",  # noqa: F822
]
