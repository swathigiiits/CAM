# Smart Helmet Attention Analytics Agent

AI-powered driver attention analysis platform using gaze tracking, video scene understanding, and coaching.

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open http://localhost:3000

## Full Feature Set (Phase 1–3)

### Processing Pipeline
- Session upload (CSV + video)
- Video + CSV timestamp synchronization
- Gaze homography calibration (4-point)
- OpenCV scene understanding (road, vehicles, pedestrians, signs, traffic lights)
- Fixation detection (I-VT)
- Scene-aware attention metrics
- Hazard reaction latency
- Environmental scanning quality
- Persistent local storage (`backend/data/`)

### Analytics
- Attention score (0–100 composite)
- Risk level & driver level (Beginner → Advanced)
- Road attention, gaze stability, attention spread
- Off-road glances & long distraction events (>2s)
- Scene distribution from video CV or gaze zones
- Fixation density + temporal heatmaps
- Session comparison with deltas
- Longitudinal progress (weekly trends, consistency, maturity)

### Coaching (Phase 3)
- Rule-based coaching report with session comparison
- Explainable insights (metric, evidence, recommendation)
- LLM coaching via Anthropic API (falls back to rules)
- Multi-session driver context in LLM prompts

### Optional Metadata
- Driver ID, weather, session type, notes
- GPS coordinates, average speed

## CSV Format

```csv
timestamp,gaze_x,gaze_y,pupil_size
0.01,512,302,0.82
0.03,510,299,0.81
```

Use `GET /demo/generate-csv` or the "Use Demo CSV" button to test.

## API Highlights

| Endpoint | Description |
|----------|-------------|
| `POST /sessions/upload` | Upload CSV + optional video |
| `POST /sessions/{id}/calibration` | Set homography (4 src/dst points) |
| `GET /sessions/{id}/video` | Stream session video |
| `GET /drivers/{id}/progress` | Longitudinal tracking |
| `POST /sessions/compare` | Multi-session comparison |
| `POST /sessions/{id}/coaching/llm` | AI coaching (rule fallback) |

## LLM Coaching

Set `ANTHROPIC_API_KEY` on the backend for LLM-powered coaching. Without it, rule-based reports are returned automatically.

## AWS Deployment

See `backend/AWS_DEPLOYMENT.md` for ECS, S3, DynamoDB setup.
