"""GridSense AI Data Simulator — publishes DTM readings for 100 transformers and 5000 smart meters."""
from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
PUBLISH_INTERVAL = float(os.getenv("PUBLISH_INTERVAL", "5.0"))
N_TRANSFORMERS = int(os.getenv("N_TRANSFORMERS", "100"))
METERS_PER_TRANSFORMER = int(os.getenv("METERS_PER_TRANSFORMER", "50"))
DEGRADING_IDS_RAW = os.getenv("DEGRADING_TRANSFORMERS", "T-047,T-023")
DEGRADING_TRANSFORMERS: list[str] = [x.strip() for x in DEGRADING_IDS_RAW.split(",")]
THEFT_METER = os.getenv("THEFT_METER", "M-04702")


class GridSimulator:
    """Simulates DTM and smart meter data for 100 transformers and 5000 meters.

    Publishes readings over MQTT to the GridSense ingestion layer.
    Certain transformers are marked as degrading to exercise the ML pipeline.
    One meter is designated as a theft meter to exercise theft detection.
    """

    def __init__(
        self,
        broker_host: str = BROKER_HOST,
        broker_port: int = BROKER_PORT,
    ) -> None:
        """Initialize transformer/meter ID lists, per-transformer state, and MQTT client."""
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._transformer_ids: list[str] = [
            self._transformer_id(n) for n in range(1, N_TRANSFORMERS + 1)
        ]
        self._meter_ids: dict[str, list[str]] = {
            self._transformer_id(t): [
                self._meter_id(t, m) for m in range(1, METERS_PER_TRANSFORMER + 1)
            ]
            for t in range(1, N_TRANSFORMERS + 1)
        }
        self._state: dict[str, dict[str, Any]] = {
            tid: {"timestep": 0} for tid in self._transformer_ids
        }

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    @staticmethod
    def _transformer_id(n: int) -> str:
        """Return a zero-padded transformer ID string.

        Args:
            n: Transformer sequence number (1-based).

        Returns:
            Formatted transformer ID, e.g. 'T-001'.
        """
        return f"T-{n:03d}"

    @staticmethod
    def _meter_id(transformer_n: int, meter_n: int) -> str:
        """Return a composite meter ID string.

        Args:
            transformer_n: Transformer sequence number (1-based).
            meter_n: Meter sequence number within transformer (1-based).

        Returns:
            Formatted meter ID, e.g. 'M-00102'.
        """
        return f"M-{transformer_n:03d}{meter_n:02d}"

    def _generate_transformer_reading(self, transformer_id: str) -> dict[str, Any]:
        """Generate a synthetic DTM sensor reading for a transformer.

        Applies additional degradation noise and drift if the transformer is in
        the DEGRADING_TRANSFORMERS list.

        Args:
            transformer_id: The transformer to generate a reading for.

        Returns:
            Dict with all 13 DTM fields plus timestamp as ISO string.
        """
        timestep = self._state[transformer_id]["timestep"]
        is_degrading = transformer_id in DEGRADING_TRANSFORMERS

        # Base readings drawn from normal distributions
        Va = float(np.random.normal(230.0, 2.0))
        Vb = float(np.random.normal(230.0, 2.0))
        Vc = float(np.random.normal(230.0, 2.0))
        Ia = float(np.random.normal(80.0, 5.0))
        Ib = float(np.random.normal(80.0, 5.0))
        Ic = float(np.random.normal(80.0, 5.0))
        oil_temp = float(np.random.normal(55.0, 3.0))
        power_factor = float(np.random.normal(0.92, 0.02))
        thd_pct = float(np.random.normal(3.5, 0.5))
        active_power_kw = float(np.random.normal(120.0, 10.0))
        reactive_power_kvar = float(np.random.normal(40.0, 5.0))
        tamper_flag = False

        if is_degrading:
            # Progressive degradation: oil temperature rises, THD increases,
            # power factor drops, and voltage sags slightly over time.
            drift = min(timestep * 0.05, 20.0)
            oil_temp += drift + float(np.random.normal(0, 1.5))
            thd_pct += drift * 0.3 + float(np.random.normal(0, 0.8))
            power_factor -= drift * 0.005 + float(np.random.normal(0, 0.01))
            Va -= drift * 0.2 + float(np.random.normal(0, 1.0))
            Vb -= drift * 0.2 + float(np.random.normal(0, 1.0))
            Vc -= drift * 0.2 + float(np.random.normal(0, 1.0))

        # Clamp power factor to valid range
        power_factor = float(np.clip(power_factor, 0.0, 1.0))

        return {
            "transformer_id": transformer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "Va": Va,
            "Vb": Vb,
            "Vc": Vc,
            "Ia": Ia,
            "Ib": Ib,
            "Ic": Ic,
            "oil_temp": oil_temp,
            "power_factor": power_factor,
            "thd_pct": thd_pct,
            "active_power_kw": active_power_kw,
            "reactive_power_kvar": reactive_power_kvar,
            "tamper_flag": tamper_flag,
            "anomaly_score": None,
        }

    def _generate_meter_reading(
        self, meter_id: str, transformer_id: str, timestep: int
    ) -> dict[str, Any]:
        """Generate a synthetic smart meter reading.

        If the meter is the designated theft meter and enough time has elapsed,
        consumption drops dramatically and the tamper flag is set.

        Args:
            meter_id: The smart meter ID.
            transformer_id: The parent transformer ID.
            timestep: Current simulation timestep for this transformer.

        Returns:
            Dict with meter reading fields and timestamp as ISO string.
        """
        active_power_kw = float(np.random.normal(2.5, 0.5))
        reactive_power_kvar = float(np.random.normal(0.8, 0.1))
        tamper_flag = False

        if meter_id == THEFT_METER and timestep >= 50:
            active_power_kw *= 0.10
            tamper_flag = True

        return {
            "meter_id": meter_id,
            "transformer_id": transformer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_power_kw": active_power_kw,
            "reactive_power_kvar": reactive_power_kvar,
            "tamper_flag": tamper_flag,
        }

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        rc: int,
        properties: Any = None,
    ) -> None:
        """Handle MQTT connection event.

        Args:
            client: The MQTT client instance.
            userdata: User-defined data (unused).
            flags: Connection flags from broker.
            rc: Return code (0 = success).
            properties: MQTT v5 properties (optional).
        """
        if rc == 0:
            logger.info(
                "GridSimulator connected to MQTT broker at %s:%d",
                self._broker_host,
                self._broker_port,
            )
        else:
            logger.error("GridSimulator failed to connect, return code: %d", rc)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        disconnect_flags: Any,
        rc: int,
        properties: Any = None,
    ) -> None:
        """Handle MQTT disconnection event.

        Args:
            client: The MQTT client instance.
            userdata: User-defined data (unused).
            disconnect_flags: Disconnect flags.
            rc: Return code.
            properties: MQTT v5 properties (optional).
        """
        logger.warning("GridSimulator disconnected from broker (rc=%d)", rc)

    def run(self) -> None:
        """Main simulation loop — publishes readings for all transformers and meters.

        Iterates over every transformer each tick, generates a DTM reading and
        readings for all attached meters, then publishes each over MQTT.
        Increments the per-transformer timestep after each tick.
        Handles KeyboardInterrupt to disconnect cleanly.
        """
        logger.info(
            "Starting simulation: %d transformers, %d meters each, interval=%.1fs",
            N_TRANSFORMERS,
            METERS_PER_TRANSFORMER,
            PUBLISH_INTERVAL,
        )
        try:
            while True:
                for transformer_num in range(1, N_TRANSFORMERS + 1):
                    t_id = self._transformer_id(transformer_num)
                    timestep = self._state[t_id]["timestep"]

                    reading = self._generate_transformer_reading(t_id)
                    payload = json.dumps(reading, default=str)
                    self._client.publish(
                        f"gridsense/transformers/{t_id}/readings", payload, qos=1
                    )

                    for meter_num in range(1, METERS_PER_TRANSFORMER + 1):
                        m_id = self._meter_id(transformer_num, meter_num)
                        m_reading = self._generate_meter_reading(m_id, t_id, timestep)
                        m_payload = json.dumps(m_reading, default=str)
                        self._client.publish(
                            f"gridsense/meters/{m_id}/readings", m_payload, qos=1
                        )

                    self._state[t_id]["timestep"] += 1

                logger.debug("Published tick for %d transformers", N_TRANSFORMERS)
                time.sleep(PUBLISH_INTERVAL)

        except KeyboardInterrupt:
            logger.info("GridSimulator shutting down on KeyboardInterrupt.")
            self._client.disconnect()

    def start(self) -> None:
        """Connect to the MQTT broker and begin the simulation loop.

        Starts the MQTT network loop in a background thread, then calls run()
        which blocks until interrupted.
        """
        self._client.connect(self._broker_host, self._broker_port)
        self._client.loop_start()
        self.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    simulator = GridSimulator()
    simulator.start()
