# metamorphicTerritory — perpetual mode roadmap

## Status
- [x] FastAPI tile server (serve.py) with 7 endpoints
- [x] tippecanoe GeoJSON -> PMTiles + XYZ tile directory
- [x] Checkpoint system: per-cell state at years 0,1,3,5,7,10,15,20
- [x] /state/{year} endpoint for timeline scrubbing
- [x] WebSocket /ws perpetual mode — live GoL diffs every 2s
- [x] /perpetual/start and /perpetual/stop controls
- [x] systemd service (metamorphic-territory.service)
- [x] Seeds from checkpoint files (survives server restart)

## Perpetual mode
The GoL engine runs continuously after seeding from the latest simulation.
Each season step takes ~1.3s. Diffs streamed to WebSocket clients.
The forest never converges — species compete, spread, and die perpetually.

### Run it
```bash
# Start service
systemctl start metamorphic-territory

# Seed + start perpetual
curl -X POST http://localhost:8420/perpetual/start

# Monitor
curl http://localhost:8420/perpetual/status

# Stop
curl -X POST http://localhost:8420/perpetual/stop
```

## Next steps

### P0 — immediate
- [ ] Auto-start perpetual mode on service boot (after loading latest checkpoint)
- [ ] Persist perpetual state to checkpoint files every 10 years of sim time
- [ ] Rate-limit WebSocket diffs to max 1/sec to reduce frontend repaints

### P1 — planting reports
- [ ] /report endpoint: generate unified planting document
- [ ] Schedule 80K moves across seasons/weeks (not all at once)
- [ ] Each stratum layer as a report section
- [ ] Export as PDF/HTML with hex maps per phase
- [ ] Species palette per cluster with guild compatibility scores

### P2 — integration
- [ ] Connect to OpenClaw for autonomous re-simulation when new LiDAR arrives
- [ ] Upload PMTiles to GCS after each simulation run
- [ ] Cloudflare Tunnel for public HTTPS (currently Tailscale only)
- [ ] CUDA acceleration for GoL on 2070 GPU

### P3 — science
- [ ] Add real environmental layers (drainage COG, CHM COG, solar) to ll0odog
- [ ] Calibrate GoL parameters against observed tree mortality/growth
- [ ] Multi-scan validation: compare predicted vs actual LiDAR deltas
- [ ] Seasonal weather modulation from climate data
