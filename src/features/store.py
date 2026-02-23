"""
src/features/store.py
---------------------
Redis-backed feature store for real-time feature retrieval.
Provides sub-5ms access to precomputed user and merchant features.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from src.core.interfaces import BaseFeatureStore

logger = logging.getLogger(__name__)

# Key prefixes
USER_FEATURES_PREFIX = "fg:user:"
MERCHANT_FEATURES_PREFIX = "fg:merchant:"
VELOCITY_PREFIX = "fg:velocity:"


class RedisFeatureStore(BaseFeatureStore):
    """
    Async Redis feature store.

    Key schema:
        fg:user:{user_id}          → JSON dict of user behavioral features
        fg:merchant:{merchant_id}  → JSON dict of merchant risk features
        fg:velocity:{user_id}:{window_minutes} → transaction count in window

    All keys have a TTL (default 24h) to prevent stale features.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db: int = 0,
        ttl_seconds: int = 86400,
        pool_size: int = 20,
    ) -> None:
        self._redis_url = redis_url
        self._db = db
        self._ttl = ttl_seconds
        self._pool_size = pool_size
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self._redis_url,
                db=self._db,
                max_connections=self._pool_size,
                decode_responses=True,
            )
        return self._client

    async def ping(self) -> bool:
        """Test Redis connectivity."""
        client = await self._get_client()
        return await client.ping()

    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            await self._client.aclose()

    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        """
        Retrieve precomputed user behavioral features.

        Args:
            user_id: Tokenized user identifier

        Returns:
            Feature dict (empty dict if user not in store)
        """
        client = await self._get_client()
        key = f"{USER_FEATURES_PREFIX}{user_id}"

        try:
            raw = await client.get(key)
            if raw is None:
                return {}
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Feature store get failed for user=%s: %s", user_id, exc)
            return {}

    async def get_merchant_features(self, merchant_id: str) -> dict[str, Any]:
        """Retrieve precomputed merchant risk features."""
        client = await self._get_client()
        key = f"{MERCHANT_FEATURES_PREFIX}{merchant_id}"

        try:
            raw = await client.get(key)
            if raw is None:
                return {}
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Feature store get failed for merchant=%s: %s", merchant_id, exc)
            return {}

    async def update_user_features(self, user_id: str, features: dict[str, Any]) -> None:
        """Update user features after a transaction (atomic get-update-set)."""
        client = await self._get_client()
        key = f"{USER_FEATURES_PREFIX}{user_id}"

        try:
            # Get existing features
            existing_raw = await client.get(key)
            existing = json.loads(existing_raw) if existing_raw else {}

            # Merge update
            merged = {**existing, **features}

            await client.set(key, json.dumps(merged), ex=self._ttl)
        except Exception as exc:
            logger.warning("Feature store update failed for user=%s: %s", user_id, exc)

    async def get_velocity_features(
        self,
        user_id: str,
        windows: list[int],
    ) -> dict[str, float]:
        """
        Get real-time transaction velocity for a user over multiple time windows.

        Args:
            user_id: Tokenized user identifier
            windows: Time windows in minutes [60, 360, 1440, ...]

        Returns:
            Dict mapping window_minutes → txn count
        """
        client = await self._get_client()
        result: dict[str, float] = {}

        for window_minutes in windows:
            key = f"{VELOCITY_PREFIX}{user_id}:{window_minutes}"
            try:
                val = await client.get(key)
                result[f"velocity_{window_minutes}m"] = float(val) if val else 0.0
            except Exception:
                result[f"velocity_{window_minutes}m"] = 0.0

        return result

    async def increment_velocity(self, user_id: str, window_minutes: int) -> None:
        """Increment transaction count for velocity tracking (called after each txn)."""
        client = await self._get_client()
        key = f"{VELOCITY_PREFIX}{user_id}:{window_minutes}"

        try:
            await client.incr(key)
            await client.expire(key, window_minutes * 60)
        except Exception as exc:
            logger.debug("Velocity increment failed: %s", exc)

    async def bulk_get_user_features(self, user_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch fetch user features using Redis pipeline for efficiency."""
        client = await self._get_client()
        keys = [f"{USER_FEATURES_PREFIX}{uid}" for uid in user_ids]

        try:
            pipe = client.pipeline(transaction=False)
            for key in keys:
                pipe.get(key)
            values = await pipe.execute()

            result = {}
            for uid, raw in zip(user_ids, values):
                result[uid] = json.loads(raw) if raw else {}
            return result
        except Exception as exc:
            logger.warning("Bulk feature fetch failed: %s", exc)
            return {uid: {} for uid in user_ids}
