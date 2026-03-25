#!/bin/bash
# ActionPlan Streaming Demo (HPC / SSH tunnel ready)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ActionPlan Streaming Motion Demo ==="
echo ""

# -----------------------------
# Python venv
# -----------------------------
#source $WORK/venvs/pytorch-env/bin/activate

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# -----------------------------
# Ports
# -----------------------------
export BACKEND_PORT=8000
export FRONTEND_PORT=3000

# -----------------------------
# Node setup
# -----------------------------
if module avail 2>&1 | grep -q node; then
    module load nodejs/18
else
    export PATH=$WORK/tools/node-v20.11.1-linux-x64/bin:$PATH
fi

if ! command -v node >/dev/null 2>&1; then
    echo "Error: node/npm not found!"
    exit 1
fi
echo "Node version: $(node -v), npm version: $(npm -v)"
echo ""

# -----------------------------
# System info
# -----------------------------
hostname
nvidia-smi || echo "No GPU detected"
python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF
echo ""

# -----------------------------
# Backend deps
# -----------------------------
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
fi

# -----------------------------
# Frontend deps
# -----------------------------
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# -----------------------------
# Patch frontend to use tunneled backend
# -----------------------------
echo "Patching frontend to use backend via localhost:$BACKEND_PORT..."
FRONTEND_ENV_FILE="frontend/.env.local"
mkdir -p frontend
cat > $FRONTEND_ENV_FILE <<EOL
VITE_BACKEND_URL=http://localhost:$BACKEND_PORT
EOL

# -----------------------------
# Build frontend (production)
# -----------------------------
echo "Building frontend (production mode)..."
cd frontend
npm run build
cd ..

# -----------------------------
# Start backend (from project root so config paths resolve)
# -----------------------------
ACTIONPLAN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Starting backend on port $BACKEND_PORT..."
cd "$ACTIONPLAN_ROOT"
python demo/server.py --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

sleep 5

# -----------------------------
# Start frontend
# -----------------------------
echo "Starting frontend (production server) on port $FRONTEND_PORT..."
cd frontend
npx serve -s dist -l $FRONTEND_PORT &
FRONTEND_PID=$!
cd ..

echo ""
echo "=== Servers running on compute node ==="
echo "Backend  : $BACKEND_PORT"
echo "Frontend : $FRONTEND_PORT"
echo ""
echo "Press Ctrl+C to stop both servers"

# -----------------------------
# Cleanup
# -----------------------------
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

wait