"""MQTT consumer — subscribes to all transformer and meter topics."""
from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from gridsense.db import store
from gridsense.schemas.readings import TransformerReadingSchema, MeterReadingSchema

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))


class MQTTConsumer:
    """MQTT subscriber that ingests transformer and smart meter readings into the store.

    Subscribes to wildcard topics for all transformers and meters, validates
    incoming payloads with Pydantic v2 schemas, and writes records to the
    thread-safe in-memory store.
    """

    def __init__(
        self,
        broker_host: str = BROKER_HOST,
        broker_port: int = BROKER_PORT,
    ) -> None:
        """Initialize the MQTT client and register callbacks."""
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        self._running: bool = False
        self._message_count: int = 0

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        rc: int,
        properties: Any = None,
    ) -> None:
        if rc == 0:
            client.subscribe("gridsense/transformers/+/readings", qos=1)
            client.subscribe("gridsense/meters/+/readings", qos=1)
            logger.info("Connected to broker at %s:%d — subscribed to all topics.", self._broker_host, self._broker_port)
        else:
            logger.error("Failed to connect to broker, return code: %d", rc)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        try:
            topic_parts = msg.topic.split("/")
            if len(topic_parts) != 4:
                logger.warning("Unexpected topic structure: %s", msg.topic)
                return

            entity_type = topic_parts[1]
            payload = json.loads(msg.payload.decode("utf-8"))

            if entity_type == "transformers":
                schema = TransformerReadingSchema.model_validate(payload)
                store.append_transformer_reading(schema.model_dump())
                self._message_count += 1
                if self._message_count % 100 == 0:
                    logger.info(
                        "Ingested %d messages | Transformers in store: %d | Meters: %d",
                        self._message_count,
                        len(store.STORE.get("transformer_readings", [])),
                        len(store.STORE.get("meter_readings", [])),
                    )
                logger.debug("Stored transformer reading for %s", schema.transformer_id)

            elif entity_type == "meters":
                schema = MeterReadingSchema.model_validate(payload)
                store.append_meter_reading(schema.model_dump())
                logger.debug("Stored meter reading for %s", schema.meter_id)

            else:
                logger.warning("Unknown entity type in topic: %s", entity_type)

        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON on topic %s: %s", msg.topic, exc)
        except Exception as exc:
            logger.error(
                "Error processing message on topic %s: %s",
                msg.topic,
                exc,
                exc_info=True,
            )

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        disconnect_flags: Any,
        rc: int,
        properties: Any = None,
    ) -> None:
        self._running = False
        if rc != 0:
            logger.warning("Unexpected disconnection (rc=%d). Will attempt reconnect...", rc)

    def start(self) -> None:
        """Connect to broker and start background loop thread (non-blocking)."""
        self._client.connect(self._broker_host, self._broker_port)
        self._client.loop_start()
        self._running = True
        logger.info("MQTTConsumer started (background thread).")

    def start_blocking(self) -> None:
        """Connect to broker and block forever — use when running as __main__."""
        logger.info("Connecting to broker at %s:%d ...", self._broker_host, self._broker_port)
        self._client.connect(self._broker_host, self._broker_port)
        self._running = True
        logger.info("MQTTConsumer running — press Ctrl+C to stop.")
        try:
            # loop_forever blocks the main thread and handles reconnects automatically
            self._client.loop_forever(retry_first_connection=True)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received — shutting down consumer.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the network loop and disconnect cleanly."""
        self._client.loop_stop()
        self._client.disconnect()
        self._running = False
        logger.info("MQTTConsumer stopped. Total messages ingested: %d", self._message_count)


def start_consumer() -> MQTTConsumer:
    """Create, start, and return an MQTTConsumer (non-blocking — for use in threads).

    Returns:
        A running MQTTConsumer connected to the configured broker.
    """
    consumer = MQTTConsumer()
    consumer.start()
    return consumer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting GridSense MQTT ingestion consumer...")
    consumer = MQTTConsumer()
    consumer.start_blocking()