#!/usr/bin/env bash
# ── metamorphic territory tile service — ll0odog deployment ──
set -euo pipefail

echo "=== 1/5  System deps ==="
sudo apt-get update
sudo apt-get install -y build-essential libsqlite3-dev zlib1g-dev curl git

# tippecanoe (GeoJSON → vector tiles)
if ! command -v tippecanoe &>/dev/null; then
    echo "Installing tippecanoe..."
    git clone --depth 1 https://github.com/felt/tippecanoe.git /tmp/tippecanoe
    cd /tmp/tippecanoe && make -j$(nproc) && sudo make install && cd -
    rm -rf /tmp/tippecanoe
    echo "tippecanoe $(tippecanoe --version 2>&1 | head -1)"
else
    echo "tippecanoe already installed"
fi

# uv (fast Python package manager)
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== 2/5  Clone / pull repo ==="
REPO_DIR="$HOME/metamorphicTerritory"
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" && git pull
else
    git clone https://github.com/Ch-i/hooke-surveyor.git /tmp/surveyor-clone
    cp -r /tmp/surveyor-clone/metamorphicTerritory "$REPO_DIR"
    # Also copy surveyor package + species_db for the engine
    cp -r /tmp/surveyor-clone/surveyor "$REPO_DIR/surveyor"
    cp -r /tmp/surveyor-clone/species_db "$REPO_DIR/species_db"
    cp -r /tmp/surveyor-clone/knowledge "$REPO_DIR/knowledge"
    rm -rf /tmp/surveyor-clone
fi
cd "$REPO_DIR"

echo "=== 3/5  Install Python deps ==="
uv sync --extra serve
echo "Installed. Testing import..."
uv run python -c "from metamorphic_territory.serve import app; print('FastAPI app OK')"

echo "=== 4/5  Run first simulation ==="
mkdir -p output

# Start the server in the background
uv run metamorphic-serve &
SERVER_PID=$!
sleep 3

# Fire the simulation (Hooke Park center, res-12, 20 years)
echo "Running simulation..."
curl -s -X POST http://localhost:8420/simulate \
    -H 'Content-Type: application/json' \
    -d '{
      "lat": 50.791,
      "lng": -2.669,
      "k_ring": 18,
      "resolution": 12,
      "years": 20,
      "seed": 42
    }' | python3 -m json.tool

# Verify tiles exist
echo ""
echo "=== Tile check ==="
ls -lh output/scheme.pmtiles output/scheme.geojson 2>/dev/null || echo "WARNING: tiles not generated"
ls output/tiles/ 2>/dev/null | head -5 && echo "..." || echo "WARNING: tile directory empty"

# Health check
echo ""
curl -s http://localhost:8420/health | python3 -m json.tool

echo ""
echo "=== 5/5  Server running on :8420 ==="
echo "PID: $SERVER_PID"
echo ""
echo "Test in browser: http://localhost:8420/health"
echo "Kill with: kill $SERVER_PID"
echo ""
echo "── Install as systemd service (auto-start on boot) ──"
echo "sudo cp metamorphic-territory.service /etc/systemd/system/"
echo "sudo systemctl daemon-reload"
echo "sudo systemctl enable --now metamorphic-territory"
echo ""
echo "── Next: set up Cloudflare Tunnel ──"
echo "1. Install cloudflared:"
echo "   curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
echo ""
echo "2. Login + create tunnel:"
echo "   cloudflared tunnel login"
echo "   cloudflared tunnel create metamorphic"
echo "   cloudflared tunnel route dns metamorphic forest.cyberneticlandscapes.com"
echo ""
echo "3. Run tunnel:"
echo "   cloudflared tunnel run --url http://localhost:8420 metamorphic"
echo ""
echo "4. Set frontend env:"
echo "   VITE_METAMORPHIC_URL=https://forest.cyberneticlandscapes.com"
