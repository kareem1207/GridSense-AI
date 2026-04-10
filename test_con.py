import paho.mqtt.client as mqtt
import json
import sys

# Import your store
try:
    from gridsense.db.store import STORE
    print(f"Store imported OK. Keys: {list(STORE.keys())}")
except Exception as e:
    print(f"Store import FAILED: {e}")
    sys.exit(1)

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to Mosquitto OK")
        client.subscribe("gridsense/#")
        print("Subscribed to gridsense/#")
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        topic = msg.topic

        if "transformers" in topic:
            STORE["transformer_readings"].append(data)
            t_id = data.get("transformer_id", "unknown")
            print(f"[TRANSFORMER] {t_id} | THD: {data.get('thd_pct', 'N/A'):.2f} | Oil: {data.get('oil_temp', 'N/A'):.1f}°C | Total readings: {len(STORE['transformer_readings'])}")

        elif "meters" in topic:
            STORE["meter_readings"].append(data)

    except Exception as e:
        print(f"Error processing message: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

print("Connecting to Mosquitto on localhost:1883...")
try:
    client.connect("127.0.0.1", 1883, 60)
    print("Starting loop — you should see transformer readings every 5 seconds...")
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Connection error: {e}")