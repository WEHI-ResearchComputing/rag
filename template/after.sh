# Wait for the Jupyter Notebook server to start
echo "Waiting for Jupyter Notebook server to open port 7860..."
echo "TIMING - Starting wait at: $(date)"
if wait_until_port_used "${host}:7860" 60; then
  echo "Discovered Jupyter Notebook server listening on port 7860"
  echo "TIMING - Wait ended at: $(date)"
else
  echo "Timed out waiting for Jupyter Notebook server to open port 7860!"
  echo "TIMING - Wait ended at: $(date)"
  pkill -P ${SCRIPT_PID}
  clean_up 1
fi
sleep 2
