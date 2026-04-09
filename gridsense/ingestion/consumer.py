"""MQTT consumer — subscribes to all transformer and meter topics."""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from gridsense.db import store
from gridsense.schemas.readings import TransformerReadingSchema, MeterReadingSchema

load_dotenv()
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

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        rc: int,
        properties: Any = None,
    ) -> None:
        """Subscribe to transformer and meter reading topics on successful connect.

        Args:
            client: The MQTT client instance.
            userdata: User-defined data (unused).
            flags: Connection flags from broker.
            rc: Return code (0 = success).
            properties: MQTT v5 properties (optional).
        """
        if rc == 0:
            client.subscribe("gridsense/transformers/+/readings", qos=1)
            client.subscribe("gridsense/meters/+/readings", qos=1)
            logger.info(
                "MQTTConsumer connected and subscribed to transformer and meter topics."
            )
        else:
            logger.error("MQTTConsumer failed to connect, return code: %d", rc)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        """Parse and store an incoming MQTT message.

        Routes transformer readings to TransformerReadingSchema and meter
        readings to MeterReadingSchema, then writes to the in-memory store.

        Args:
            client: The MQTT client instance.
            userdata: User-defined data (unused).
            msg: The received MQTT message.
        """
        try:
            topic_parts = msg.topic.split("/")
            # Expected formats:
            #   gridsense/transformers/<transformer_id>/readings  (4 parts)
            #   gridsense/meters/<meter_id>/readings              (4 parts)
            if len(topic_parts) != 4:
                logger.warning("Unexpected topic structure: %s", msg.topic)
                return

            entity_type = topic_parts[1]
            payload = json.loads(msg.payload.decode("utf-8"))

            if entity_type == "transformers":
                schema = TransformerReadingSchema.model_validate(payload)
                store.append_transformer_reading(schema.model_dump())
                logger.debug(
                    "Stored transformer reading for %s", schema.transformer_id
                )
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
                "Error processing message on topic %s: %s", msg.topic, exc, exc_info=True
            )

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        disconnect_flags: Any,
        rc: int,
        properties: Any = None,
    ) -> None:
        """Handle disconnection from the MQTT broker.

        Args:
            client: The MQTT client instance.
            userdata: User-defined data (unused).
            disconnect_flags: Disconnect flags.
            rc: Return code.
            properties: MQTT v5 properties (optional).
        """
        self._running = False
        logger.warning(
            "MQTTConsumer disconnected from broker (rc=%d). Reconnect may be needed.", rc
        )

    def start(self) -> None:
        """Connect to the MQTT broker and start the background network loop."""
        self._client.connect(self._broker_host, self._broker_port)
        self._client.loop_start()
        self._running = True
        logger.info(
            "MQTTConsumer started, connected to %s:%d",
            self._broker_host,
            self._broker_port,
        )

    def stop(self) -> None:
        """Stop the network loop and disconnect from the broker."""
        self._client.loop_stop()
        self._client.disconnect()
        self._running = False
        logger.info("MQTTConsumer stopped.")


def start_consumer() -> MQTTConsumer:
    """Create, start, and return an MQTTConsumer instance.

    Returns:
        A running MQTTConsumer connected to the configured broker.
    """
    consumer = MQTTConsumer()
    consumer.start()
    return consumer
