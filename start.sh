#!/bin/bash
echo "============================================"
echo "       GridSense AI — Starting Up          "
echo "============================================"

echo ""
echo "[1/5] Starting Mosquitto MQTT Broker..."
"E:\Mosquitto\mosquitto.exe" -v -c "E:\Mosquitto\mosquitto.conf" &
sleep 2
echo "      Mosquitto running on port 1883"

echo ""
echo "[2/5] Starting Gemma 4 LLM Server..."
"E:\llm\bin\llama-server.exe" \
  -m "E:\llm\models\gemma4-e4b\google_gemma-4-E4B-it-Q4_K_M.gguf" \
  --port 8080 --ctx-size 4096 --n-gpu-layers 0 &
sleep 5
echo "      LLM running on http://127.0.0.1:8080"

echo ""
echo "[3/5] Starting Simulator..."
python -m gridsense.simulator.main &
sleep 2
echo "      Simulator publishing 100 transformers"

echo ""
echo "[4/5] Starting FastAPI..."
uvicorn gridsense.api.main:app --port 8000 --reload &
sleep 2
echo "      API running on http://127.0.0.1:8000"

echo ""
echo "[5/5] Starting Dashboard..."
streamlit run gridsense/dashboard/app.py &
echo "      Dashboard on http://localhost:8501"

echo ""
echo "============================================"
echo "  All systems running! Open your browser:"
echo "  Dashboard : http://localhost:8501"
echo "  API Docs  : http://localhost:8000/docs"
echo "  LLM       : http://127.0.0.1:8080"
echo "============================================"

# Keep script running
wait