import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import litellm
from litellm.types.utils import ModelInfo

OPENAI_MAX_PROMPT_CACHE_KEY_LENGTH: Final = 64

_EFFORT_DEGRADATION_CHAIN: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "max": ("max", "xhigh", "high"),
        "xhigh": ("xhigh", "high"),
        "minimal": ("minimal", "low"),
    }
)
_UNCONDITIONALLY_ACCEPTED_EFFORT: Final = "medium"


def prompt_cache_key_from_user_id(user_id: object) -> str | None:
    if user_id is None:
        return None
    return str(user_id)[:OPENAI_MAX_PROMPT_CACHE_KEY_LENGTH] or None


def local_model_name(model: str, custom_llm_provider: object) -> str:
    """The id the provider itself knows, for reporting back to the caller in ``message_start``."""
    return model.removeprefix(f"{custom_llm_provider}/") if isinstance(custom_llm_provider, str) else model


def is_reasoning_auto_summary_enabled() -> bool:
    """Check whether the default 'summary: detailed' injection is enabled (opt-in)."""
    return litellm.reasoning_auto_summary or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"


def normalize_reasoning_effort_value(
    effort: str,
    model: str,
    custom_llm_provider: str | None = None,
) -> str:
    """Lower a tier the deployment does not accept to the nearest one it does, leaving others alone.

    The accepted set is resolved by the same owner that answers ``/model_group/info``, so a level
    the proxy advertises is a level this path forwards. A deployment the map says nothing about
    keeps the chain's floor, which is what it degraded to before there was anything to ask.

    A deployment that refuses every step of a chain lands on ``medium``, which the resolver holds
    unconditional for any reasoning model: gpt-5.5-pro sets ``supports_low_reasoning_effort`` false,
    so ``minimal`` there resolves to ``medium`` rather than to a level the map already rejected.
    ``none`` is deliberately absent, being an off switch rather than a tier; an always-on-thinking
    model is handled where the thinking block is built, not by lifting the caller to a lower tier.
    """
    chain: Final = _EFFORT_DEGRADATION_CHAIN.get(effort)
    if chain is None:
        return effort

    from litellm.router_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts
    from litellm.utils import get_model_info

    try:
        model_info: Final[ModelInfo] = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        return chain[-1]

    supported: Final = resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True)
    if not supported:
        return chain[-1]
    return next((level for level in chain if level in supported), _UNCONDITIONALLY_ACCEPTED_EFFORT)
