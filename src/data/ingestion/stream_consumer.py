"""
src/data/ingestion/stream_consumer.py
---------------------------------------
Async Kafka consumer for real-time transaction event processing.
Consumes from 'transactions.raw' topic and publishes predictions to 'fraud.predictions'.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class TransactionStreamConsumer:
    """
    Async Kafka consumer for real-time fraud detection.

    Consumes raw transaction events from Kafka, runs inference,
    and publishes fraud predictions back to Kafka.

    Usage:
        consumer = TransactionStreamConsumer()
        async for batch in consumer.consume(batch_size=100):
            predictions = model.predict(batch)
            await producer.publish(predictions)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._consumer = None
        self._producer = None
        self._running = False

    async def start(self) -> None:
        """Initialize Kafka consumer and producer."""
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

            kafka_config = self._settings.kafka
            self._consumer = AIOKafkaConsumer(
                kafka_config.topic_transactions,
                bootstrap_servers=kafka_config.bootstrap_servers,
                group_id=kafka_config.consumer_group,
                auto_offset_reset=kafka_config.auto_offset_reset,
                enable_auto_commit=False,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )

            self._producer = AIOKafkaProducer(
                bootstrap_servers=kafka_config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                compression_type="lz4",
            )

            await self._consumer.start()
            await self._producer.start()
            self._running = True
            logger.info("Kafka consumer started on topic: %s", kafka_config.topic_transactions)

        except Exception as exc:
            logger.error("Failed to start Kafka consumer: %s", exc)
            raise

    async def stop(self) -> None:
        """Gracefully stop the consumer."""
        self._running = False
        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            await self._producer.stop()
        logger.info("Kafka consumer stopped")

    async def consume_batches(
        self,
        batch_size: int = 100,
        timeout_ms: int = 1000,
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        """
        Async generator that yields batches of transaction messages.

        Args:
            batch_size: Maximum number of messages per batch
            timeout_ms: Max wait time for batch to fill (milliseconds)

        Yields:
            List of transaction dicts (up to batch_size)
        """
        if not self._running:
            msg = "Consumer not started. Call start() first."
            raise RuntimeError(msg)

        batch: list[dict[str, Any]] = []

        async for msg in self._consumer:
            try:
                transaction = msg.value
                batch.append(transaction)

                if len(batch) >= batch_size:
                    yield batch
                    await self._consumer.commit()
                    batch = []

            except Exception as exc:
                logger.error("Error processing message offset=%d: %s", msg.offset, exc)
                # Continue consuming — don't let one bad message stop the pipeline
                continue

        # Yield remaining messages
        if batch:
            yield batch
            await self._consumer.commit()

    async def publish_predictions(
        self,
        predictions: list[dict[str, Any]],
    ) -> None:
        """Publish fraud predictions to the output Kafka topic."""
        if not self._producer:
            return

        topic = self._settings.kafka.topic_predictions
        for prediction in predictions:
            try:
                await self._producer.send(topic, value=prediction)
            except Exception as exc:
                logger.error("Failed to publish prediction: %s", exc)

    async def publish_alert(self, alert: dict[str, Any]) -> None:
        """Publish high-risk fraud alert to alerts topic."""
        if not self._producer:
            return

        try:
            await self._producer.send(
                self._settings.kafka.topic_alerts,
                value=alert,
            )
        except Exception as exc:
            logger.error("Failed to publish alert: %s", exc)
