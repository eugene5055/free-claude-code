"""Provider quota tracking - queries each provider's quota API."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG, ProviderDescriptor
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonValue

logger = logging.getLogger(__name__)


def string_setting(settings: Settings, attr_name: str | None) -> str | None:
    """An optional string setting (as providers.runtime.config.string_setting, without crossing into providers)."""
    if attr_name is None:
        return None
    value = getattr(settings, attr_name, None)
    return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True)
class QuotaWindow:
    """A single quota window (weekly, 5-hour, etc.)."""
    label: str          # "weekly", "5_hour", "monthly", etc.
    used_pct: float     # 0-100, percentage used
    remaining_pct: float  # 0-100, percentage remaining
    resets_at: float | None = None  # epoch seconds when window resets
    limit_type: str = "quota"  # "quota" or "user_stop"
    raw_used: float | None = None  # raw used value from provider
    raw_limit: float | None = None  # raw limit value from provider

    @property
    def refreshes_in(self) -> str | None:
        """Human-readable time until reset."""
        if self.resets_at is None:
            return None
        delta = self.resets_at - time.time()
        if delta <= 0:
            return "resetting"
        hours = int(delta // 3600)
        minutes = int((delta % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"


@dataclass(frozen=True, slots=True)
class ProviderQuota:
    """Quota information for a single provider."""
    provider_id: str
    display_name: str
    group: str              # "gemini", "claude", "openai", "codex", "other"
    windows: tuple[QuotaWindow, ...] = ()
    error: str | None = None
    last_checked: float = field(default_factory=time.time)

    @property
    def has_quota(self) -> bool:
        return len(self.windows) > 0 and self.error is None

    @property
    def primary_window(self) -> QuotaWindow | None:
        """Return the most relevant window (weekly preferred, then 5-hour)."""
        if not self.windows:
            return None
        # Prefer weekly, then 5_hour, then first
        for w in self.windows:
            if w.label.lower() in ("weekly", "7d", "week"):
                return w
        for w in self.windows:
            if w.label.lower() in ("5_hour", "5h", "five_hour"):
                return w
        return self.windows[0]


@dataclass
class QuotaGroup:
    """A group of providers with their quotas."""
    name: str
    display_name: str
    providers: tuple[ProviderQuota, ...]
    models: tuple[str, ...] = ()


class ProviderQuotaService:
    """Service for fetching and caching provider quota information."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: dict[str, ProviderQuota] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_fetch: dict[str, float] = {}

    def _is_cache_valid(self, provider_id: str) -> bool:
        last = self._last_fetch.get(provider_id, 0)
        return (time.time() - last) < self._cache_ttl

    def _get_provider_keys(self, descriptor: ProviderDescriptor) -> list[str]:
        """Get API keys for a provider from settings."""
        if descriptor.credential_attr is None:
            return []
        keys_attr = getattr(self.settings, descriptor.credential_attr, None)
        if not keys_attr:
            return []
        # Handle list, tuple (ApiKeyList), and dict formats
        if isinstance(keys_attr, (list, tuple)):
            return [str(k.get_secret_value() if hasattr(k, 'get_secret_value') else k) for k in keys_attr if k]
        return []

    def _get_base_url(self, descriptor: ProviderDescriptor) -> str | None:
        """Get the base URL for a provider."""
        if descriptor.base_url_attr:
            return string_setting(self.settings, descriptor.base_url_attr) or descriptor.default_base_url
        return descriptor.default_base_url

    def _get_proxy(self, descriptor: ProviderDescriptor) -> str | None:
        """Get proxy for a provider."""
        if descriptor.proxy_attr:
            return string_setting(self.settings, descriptor.proxy_attr)
        return None

    def _group_for_provider(self, provider_id: str) -> str:
        """Determine the display group for a provider."""
        # Map provider IDs to display groups
        group_map = {
            "gemini": "gemini",
            "vertex": "gemini",
            "anthropic": "claude",
            "openai": "openai",
            "openai_api": "openai",
            "open_router": "openai",
            "github_copilot": "claude",
            "xai": "other",
            "groq": "other",
            "together": "other",
            "deepinfra": "other",
            "nebius": "other",
            "siliconflow": "other",
            "fireworks": "other",
            "novita": "other",
            "cerebras": "other",
            "sambanova": "other",
            "chutes": "other",
            "featherless": "other",
            "agnes": "other",
            "zenmux": "other",
            "wandb": "other",
            "azure_openai": "openai",
            "deepseek": "other",
            "mistral": "other",
            "mistral_codestral": "other",
            "opencode_zen": "other",
            "opencode_go": "other",
            "vercel": "other",
            "bedrock": "other",
            "huggingface": "other",
            "cohere": "other",
            "wafer": "other",
            "kimi": "other",
            "kimi_code": "other",
            "kilo": "other",
            "minimax": "other",
            "nvidia_nim": "other",
            "cloudflare": "other",
            "zai": "other",
            "zai_api": "other",
            "groq": "other",
            "openrouter": "openai",
            "mistral": "other",
            "alibaba": "other",
            "alibaba_cloud": "other",
            "qwen": "other",
            "tokenrouter": "other",
            "nararoute": "other",
            "poolside": "other",
            "llm7": "other",
            "scaleway": "other",
            "lightning": "other",
            "experiential": "other",
            "cheaperinference": "other",
            "orcarouter": "other",
            "atlascloud": "other",
            "xkiro": "other",
            "opper": "other",
            "ollama_cloud": "local",
            "lmstudio": "local",
            "llamacpp": "local",
            "ollama": "local",
        }
        return group_map.get(provider_id, "other")

    async def _fetch_anthropic_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Anthropic quota from /v1/organizations/{org}/usage."""
        try:
            workspace_id = self.settings.anthropic_workspace_id
            if not workspace_id:
                return ()

            headers = {
                "Accept": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }

            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/v1/organizations/{workspace_id}/usage",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                windows = []
                for item in data.get("data", []):
                    # Anthropic usage API returns usage windows
                    name = item.get("name", "")
                    used = item.get("usage", 0)
                    limit = item.get("limit", 0)
                    resets_at = item.get("resets_at")

                    if limit > 0:
                        used_pct = (used / limit) * 100
                        remaining_pct = max(0, 100 - used_pct)
                        windows.append(QuotaWindow(
                            label=name,
                            used_pct=used_pct,
                            remaining_pct=remaining_pct,
                            resets_at=resets_at,
                            raw_used=used,
                            raw_limit=limit,
                        ))
                return tuple(windows)
        except Exception:
            pass
        return ()

    async def _fetch_openai_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch OpenAI quota from /v1/usage."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/v1/usage",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                windows = []
                for item in data.get("data", []):
                    name = item.get("name", "")
                    used = item.get("usage", 0)
                    limit = item.get("limit", 0)
                    resets_at = item.get("resets_at")

                    if limit > 0:
                        used_pct = (used / limit) * 100
                        remaining_pct = max(0, 100 - used_pct)
                        windows.append(QuotaWindow(
                            label=name,
                            used_pct=used_pct,
                            remaining_pct=remaining_pct,
                            resets_at=resets_at,
                            raw_used=used,
                            raw_limit=limit,
                        ))
                return tuple(windows)
        except Exception:
            pass
        return ()

    async def _fetch_google_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Google Gemini quota from rate limit headers on /models call."""
        try:
            headers = {"x-goog-api-key": api_key}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/models?pageSize=1",
                    headers=headers,
                )

            # Check rate limit headers
            windows = []
            # Google uses x-ratelimit-limit-requests, x-ratelimit-remaining-requests, etc.
            for window_type in ["day", "minute"]:
                limit_header = f"x-ratelimit-limit-requests-{window_type}"
                remaining_header = f"x-ratelimit-remaining-requests-{window_type}"
                reset_header = f"x-ratelimit-reset-requests-{window_type}"

                if limit_header in response.headers:
                    limit = int(response.headers[limit_header])
                    remaining = int(response.headers.get(remaining_header, 0))
                    if limit > 0:
                        used_pct = ((limit - remaining) / limit) * 100
                        remaining_pct = (remaining / limit) * 100
                        reset_at = None
                        if reset_header in response.headers:
                            reset_at = float(response.headers[reset_header])
                        windows.append(QuotaWindow(
                            label=f"daily" if window_type == "day" else "per_minute",
                            used_pct=used_pct,
                            remaining_pct=remaining_pct,
                            resets_at=reset_at,
                            raw_used=limit - remaining,
                            raw_limit=limit,
                        ))
            return tuple(windows)
        except Exception:
            pass
        return ()

    async def _fetch_xkiro_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch xKiro quota from /usage."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/usage",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                windows = []
                for window in data.get("windows", []):
                    name = window.get("name", "")
                    used = window.get("used", 0)
                    limit = window.get("limit", 0)
                    resets_at = window.get("resets_at")

                    if limit > 0:
                        used_pct = (used / limit) * 100
                        remaining_pct = max(0, 100 - used_pct)
                        windows.append(QuotaWindow(
                            label=name,
                            used_pct=used_pct,
                            remaining_pct=remaining_pct,
                            resets_at=resets_at,
                            raw_used=used,
                            raw_limit=limit,
                        ))
                return tuple(windows)
        except Exception:
            pass
        return ()

    async def _fetch_vercel_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Vercel AI Gateway quota from /credits."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/credits",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                balance = data.get("balance")
                if isinstance(balance, (int, float)):
                    # Vercel shows balance, not percentage - treat as monthly
                    return (QuotaWindow(
                        label="monthly",
                        used_pct=0,  # We don't know the limit
                        remaining_pct=100 if balance > 0 else 0,
                        raw_used=balance,
                        raw_limit=None,
                    ),)
        except Exception:
            pass
        return ()

    async def _fetch_deepseek_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch DeepSeek quota from /user/balance."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/user/balance",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                balance = data.get("balance_infos", [])
                windows = []
                for item in balance:
                    currency = item.get("currency", "")
                    total = item.get("total_balance", 0)
                    available = item.get("available_balance", 0)
                    if total > 0:
                        used_pct = ((total - available) / total) * 100
                        remaining_pct = (available / total) * 100
                        windows.append(QuotaWindow(
                            label=currency,
                            used_pct=used_pct,
                            remaining_pct=remaining_pct,
                            raw_used=total - available,
                            raw_limit=total,
                        ))
                return tuple(windows)
        except Exception:
            pass
        return ()

    async def _fetch_kimi_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Kimi quota from /users/me/balance."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/users/me/balance",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                balance_data = data.get("data", {})
                available = balance_data.get("available_balance")
                if isinstance(available, (int, float)):
                    return (QuotaWindow(
                        label="balance",
                        used_pct=0,
                        remaining_pct=100 if available > 0 else 0,
                        raw_used=available,
                        raw_limit=None,
                    ),)
        except Exception:
            pass
        return ()

    async def _fetch_siliconflow_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch SiliconFlow quota from /user/info."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/user/info",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                if data.get("code") == 20000 and data.get("status") is True:
                    user_data = data.get("data", {})
                    # SiliconFlow might have balance/quota info
                    # This is provider-specific - adjust based on actual response
                    pass
        except Exception:
            pass
        return ()

    async def _fetch_wafer_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Wafer quota from /usage/me."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/usage/me?period=1d&endpoint=pass.wafer.ai",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                user_id = data.get("user_id")
                if user_id:
                    # Wafer returns usage data
                    pass
        except Exception:
            pass
        return ()

    async def _fetch_groq_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Groq quota from /usage."""
        logger.debug("Fetching Groq quota from %s", base_url)
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/usage",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                logger.debug("Groq quota response: %s", data)
                windows = []
                # Groq returns usage data with windows
                for item in data.get("data", []):
                    name = item.get("name", "")
                    used = item.get("used", 0)
                    limit = item.get("limit", 0)
                    resets_at = item.get("resets_at")

                    if limit > 0:
                        used_pct = (used / limit) * 100
                        remaining_pct = max(0, 100 - used_pct)
                        windows.append(QuotaWindow(
                            label=name,
                            used_pct=used_pct,
                            remaining_pct=remaining_pct,
                            resets_at=resets_at,
                            raw_used=used,
                            raw_limit=limit,
                        ))
                return tuple(windows)
            else:
                logger.debug("Groq quota error: %s - %s", response.status_code, response.text)
        except Exception as e:
            logger.debug("Groq quota exception: %s", e)
        return ()

    async def _fetch_openrouter_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch OpenRouter quota from /credits or /usage."""
        logger.debug("Fetching OpenRouter quota from %s", base_url)
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                # Try credits endpoint first
                response = await client.get(
                    f"{base_url.rstrip('/')}/credits",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                logger.debug("OpenRouter credits response: %s", data)
                # OpenRouter credits returns balance info: {"data": {"total_credits": 10, "total_usage": 9.51}}
                if "data" in data and isinstance(data["data"], dict):
                    total_credits = data["data"].get("total_credits", 0)
                    total_usage = data["data"].get("total_usage", 0)
                    if total_credits > 0:
                        used_pct = (total_usage / total_credits) * 100
                        remaining_pct = max(0, 100 - used_pct)
                        return (QuotaWindow(
                            label="monthly_credits",
                            used_pct=used_pct,
                            remaining_pct=remaining_pct,
                            raw_used=total_usage,
                            raw_limit=total_credits,
                        ),)

            # Try usage endpoint
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/usage",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                logger.debug("OpenRouter usage response: %s", data)
                windows = []
                for item in data.get("data", []):
                    name = item.get("name", item.get("model", ""))
                    used = item.get("usage", item.get("cost", 0))
                    limit = item.get("limit", 0)
                    resets_at = item.get("resets_at")

                    if limit > 0:
                        used_pct = (used / limit) * 100
                        remaining_pct = max(0, 100 - used_pct)
                        windows.append(QuotaWindow(
                            label=name,
                            used_pct=used_pct,
                            remaining_pct=remaining_pct,
                            resets_at=resets_at,
                            raw_used=used,
                            raw_limit=limit,
                        ))
                if windows:
                    return tuple(windows)
        except Exception as e:
            logger.debug("OpenRouter quota exception: %s", e)
        return ()

    async def _fetch_mistral_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Mistral quota from /usage or /v1/usage."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                # Try standard usage endpoint
                for endpoint in ["/usage", "/v1/usage", "/v1/organizations/usage"]:
                    try:
                        response = await client.get(
                            f"{base_url.rstrip('/')}{endpoint}",
                            headers=headers,
                        )
                        if response.is_success:
                            data = response.json()
                            windows = []
                            for item in data.get("data", []):
                                name = item.get("name", "")
                                used = item.get("usage", 0)
                                limit = item.get("limit", 0)
                                resets_at = item.get("resets_at")

                                if limit > 0:
                                    used_pct = (used / limit) * 100
                                    remaining_pct = max(0, 100 - used_pct)
                                    windows.append(QuotaWindow(
                                        label=name,
                                        used_pct=used_pct,
                                        remaining_pct=remaining_pct,
                                        resets_at=resets_at,
                                        raw_used=used,
                                        raw_limit=limit,
                                    ))
                            if windows:
                                return tuple(windows)
                    except Exception:
                        continue
        except Exception:
            pass
        return ()

    async def _fetch_cloudflare_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Cloudflare Workers AI quota from /accounts/{account_id}/ai/usage."""
        try:
            # Need account ID from settings
            account_id = getattr(self.settings, "cloudflare_account_id", None)
            if not account_id:
                return ()

            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/accounts/{account_id}/ai/usage",
                    headers=headers,
                )

            if response.is_success:
                data = response.json()
                windows = []
                # Cloudflare returns usage data
                for item in data.get("result", []):
                    name = item.get("model", item.get("name", ""))
                    used = item.get("usage", item.get("tokens_used", 0))
                    limit = item.get("limit", 0)
                    resets_at = item.get("resets_at")

                    if limit > 0:
                        used_pct = (used / limit) * 100
                        remaining_pct = max(0, 100 - used_pct)
                        windows.append(QuotaWindow(
                            label=name,
                            used_pct=used_pct,
                            remaining_pct=remaining_pct,
                            resets_at=resets_at,
                            raw_used=used,
                            raw_limit=limit,
                        ))
                if windows:
                    return tuple(windows)
        except Exception:
            pass
        return ()

    async def _fetch_zai_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Z.ai (GLM) quota from /user/balance or /usage."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                # Try balance endpoint
                for endpoint in ["/user/balance", "/v1/user/balance", "/usage", "/v1/usage"]:
                    try:
                        response = await client.get(
                            f"{base_url.rstrip('/')}{endpoint}",
                            headers=headers,
                        )
                        if response.is_success:
                            data = response.json()
                            windows = []
                            # Check for balance format
                            if "balance" in data:
                                balance = data.get("balance", 0)
                                total = data.get("total", data.get("limit", 0))
                                if total > 0:
                                    used = total - balance
                                    used_pct = (used / total) * 100
                                    remaining_pct = (balance / total) * 100
                                    windows.append(QuotaWindow(
                                        label="balance",
                                        used_pct=used_pct,
                                        remaining_pct=remaining_pct,
                                        raw_used=used,
                                        raw_limit=total,
                                    ))
                            # Check for usage windows
                            for item in data.get("data", data.get("windows", [])):
                                name = item.get("name", "")
                                used = item.get("used", 0)
                                limit = item.get("limit", 0)
                                resets_at = item.get("resets_at")

                                if limit > 0:
                                    used_pct = (used / limit) * 100
                                    remaining_pct = max(0, 100 - used_pct)
                                    windows.append(QuotaWindow(
                                        label=name,
                                        used_pct=used_pct,
                                        remaining_pct=remaining_pct,
                                        resets_at=resets_at,
                                        raw_used=used,
                                        raw_limit=limit,
                                    ))
                            if windows:
                                return tuple(windows)
                    except Exception:
                        continue
        except Exception:
            pass
        return ()

    async def _fetch_alibaba_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch Alibaba Cloud (Qwen) quota from /usage or /dashscope/usage."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                # Try various endpoints
                for endpoint in ["/usage", "/v1/usage", "/dashscope/usage", "/api/v1/usage"]:
                    try:
                        response = await client.get(
                            f"{base_url.rstrip('/')}{endpoint}",
                            headers=headers,
                        )
                        if response.is_success:
                            data = response.json()
                            windows = []
                            for item in data.get("data", data.get("windows", [])):
                                name = item.get("name", "")
                                used = item.get("used", 0)
                                limit = item.get("limit", 0)
                                resets_at = item.get("resets_at")

                                if limit > 0:
                                    used_pct = (used / limit) * 100
                                    remaining_pct = max(0, 100 - used_pct)
                                    windows.append(QuotaWindow(
                                        label=name,
                                        used_pct=used_pct,
                                        remaining_pct=remaining_pct,
                                        resets_at=resets_at,
                                        raw_used=used,
                                        raw_limit=limit,
                                    ))
                            if windows:
                                return tuple(windows)
                    except Exception:
                        continue
        except Exception:
            pass
        return ()

    async def _fetch_nvidia_nim_quota(self, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Fetch NVIDIA NIM quota - may not have public quota API."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                # Try common endpoints
                for endpoint in ["/usage", "/v1/usage", "/quota", "/billing/usage"]:
                    try:
                        response = await client.get(
                            f"{base_url.rstrip('/')}{endpoint}",
                            headers=headers,
                        )
                        if response.is_success:
                            data = response.json()
                            windows = []
                            for item in data.get("data", []):
                                name = item.get("name", "")
                                used = item.get("used", 0)
                                limit = item.get("limit", 0)
                                resets_at = item.get("resets_at")

                                if limit > 0:
                                    used_pct = (used / limit) * 100
                                    remaining_pct = max(0, 100 - used_pct)
                                    windows.append(QuotaWindow(
                                        label=name,
                                        used_pct=used_pct,
                                        remaining_pct=remaining_pct,
                                        resets_at=resets_at,
                                        raw_used=used,
                                        raw_limit=limit,
                                    ))
                            if windows:
                                return tuple(windows)
                    except Exception:
                        continue
        except Exception:
            pass
        return ()

    async def _fetch_generic_quota(self, provider_id: str, api_key: str, base_url: str) -> tuple[QuotaWindow, ...]:
        """Generic quota fetch - try common endpoints."""
        # Try common quota endpoints
        endpoints = ["/usage", "/quota", "/billing/usage", "/account/usage", "/me/usage"]

        for endpoint in endpoints:
            try:
                headers = {"Authorization": f"Bearer {api_key}"}
                async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
                    response = await client.get(
                        f"{base_url.rstrip('/')}{endpoint}",
                        headers=headers,
                    )

                if response.is_success:
                    data = response.json()
                    # Try to parse common formats
                    windows = []
                    for item in data.get("data", data.get("windows", data.get("usage", []))):
                        if isinstance(item, dict):
                            name = item.get("name", item.get("label", endpoint))
                            used = item.get("used", item.get("usage", 0))
                            limit = item.get("limit", item.get("quota", 0))
                            resets_at = item.get("resets_at", item.get("reset_at"))

                            if limit > 0:
                                used_pct = (used / limit) * 100
                                remaining_pct = max(0, 100 - used_pct)
                                windows.append(QuotaWindow(
                                    label=name,
                                    used_pct=used_pct,
                                    remaining_pct=remaining_pct,
                                    resets_at=resets_at,
                                    raw_used=used,
                                    raw_limit=limit,
                                ))
                    if windows:
                        return tuple(windows)
            except Exception:
                continue
        return ()

    async def fetch_provider_quota(self, provider_id: str) -> ProviderQuota:
        """Fetch quota for a single provider."""
        descriptor = PROVIDER_CATALOG.get(provider_id)
        if not descriptor:
            return ProviderQuota(
                provider_id=provider_id,
                display_name=provider_id,
                group="other",
                error="Unknown provider",
            )

        # Check cache
        if self._is_cache_valid(provider_id) and provider_id in self._cache:
            return self._cache[provider_id]

        keys = self._get_provider_keys(descriptor)
        if not keys:
            return ProviderQuota(
                provider_id=provider_id,
                display_name=descriptor.display_name,
                group=self._group_for_provider(provider_id),
                error="Not configured",
            )

        base_url = self._get_base_url(descriptor)
        if not base_url:
            return ProviderQuota(
                provider_id=provider_id,
                display_name=descriptor.display_name,
                group=self._group_for_provider(provider_id),
                error="No base URL",
            )

        proxy = self._get_proxy(descriptor)

        # Try each key until one works
        windows = ()
        error = None
        for key in keys:
            try:
                if provider_id == "anthropic":
                    windows = await self._fetch_anthropic_quota(key, base_url)
                elif provider_id in ("openai", "openai_api", "azure_openai"):
                    windows = await self._fetch_openai_quota(key, base_url)
                elif provider_id in ("gemini", "vertex"):
                    windows = await self._fetch_google_quota(key, base_url)
                elif provider_id == "xkiro":
                    windows = await self._fetch_xkiro_quota(key, base_url)
                elif provider_id == "vercel":
                    windows = await self._fetch_vercel_quota(key, base_url)
                elif provider_id == "deepseek":
                    windows = await self._fetch_deepseek_quota(key, base_url)
                elif provider_id in ("kimi", "kimi_code"):
                    windows = await self._fetch_kimi_quota(key, base_url)
                elif provider_id == "siliconflow":
                    windows = await self._fetch_siliconflow_quota(key, base_url)
                elif provider_id == "wafer":
                    windows = await self._fetch_wafer_quota(key, base_url)
                elif provider_id == "groq":
                    windows = await self._fetch_groq_quota(key, base_url)
                elif provider_id == "open_router":
                    windows = await self._fetch_openrouter_quota(key, base_url)
                elif provider_id in ("mistral", "mistral_codestral"):
                    windows = await self._fetch_mistral_quota(key, base_url)
                elif provider_id == "cloudflare":
                    windows = await self._fetch_cloudflare_quota(key, base_url)
                elif provider_id in ("zai", "zai_api"):
                    windows = await self._fetch_zai_quota(key, base_url)
                elif provider_id in ("alibaba_cloud", "qwencloud", "qwencloud_coding"):
                    windows = await self._fetch_alibaba_quota(key, base_url)
                elif provider_id == "nvidia_nim":
                    windows = await self._fetch_nvidia_nim_quota(key, base_url)
                else:
                    windows = await self._fetch_generic_quota(provider_id, key, base_url)

                if windows:
                    break
            except Exception as e:
                error = str(e)
                continue

        quota = ProviderQuota(
            provider_id=provider_id,
            display_name=descriptor.display_name,
            group=self._group_for_provider(provider_id),
            windows=windows,
            error=error if not windows else None,
        )

        self._cache[provider_id] = quota
        self._last_fetch[provider_id] = time.time()
        return quota

    async def get_all_quotas(self) -> dict[str, QuotaGroup]:
        """Fetch quotas for all configured providers, grouped by type."""
        # Get list of configured providers
        configured_providers = []
        for provider_id, descriptor in PROVIDER_CATALOG.items():
            keys = self._get_provider_keys(descriptor)
            if keys:
                configured_providers.append(provider_id)

        # Fetch quotas concurrently
        quotas = await asyncio.gather(*[
            self.fetch_provider_quota(pid) for pid in configured_providers
        ], return_exceptions=True)

        # Group by type
        groups: dict[str, list[ProviderQuota]] = {}
        group_models: dict[str, set[str]] = {}

        for i, quota in enumerate(quotas):
            if isinstance(quota, Exception):
                continue
            if quota.group not in groups:
                groups[quota.group] = []
                group_models[quota.group] = set()
            groups[quota.group].append(quota)
            # Collect model names from provider catalog
            desc = PROVIDER_CATALOG.get(quota.provider_id)
            if desc:
                group_models[quota.group].add(desc.display_name)

        # Build group objects
        group_display_names = {
            "gemini": "GEMINI MODELS",
            "claude": "CLAUDE AND GPT MODELS",
            "openai": "OPENAI MODELS",
            "codex": "CODEX MODELS",
            "local": "LOCAL MODELS",
            "other": "OTHER PROVIDERS",
        }

        result = {}
        for group_id, providers in groups.items():
            result[group_id] = QuotaGroup(
                name=group_id,
                display_name=group_display_names.get(group_id, group_id.upper()),
                providers=tuple(providers),
                models=tuple(sorted(group_models.get(group_id, set()))),
            )

        return result

    def clear_cache(self) -> None:
        """Clear the quota cache."""
        self._cache.clear()
        self._last_fetch.clear()
