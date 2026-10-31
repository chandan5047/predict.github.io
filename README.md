# NBA Player Prediction Dashboard

An interactive NBA player-game prediction site with a Python pipeline for points, rebounds and three-pointers made.

## Architecture
- `index.html`: static GitHub Pages dashboard; reads `data/predictions.json`.
- `pipeline.py`: collects NBA game logs, computes leakage-free lagged player features, trains ridge models on chronological data, forecasts upcoming scheduled matchups.
- `.github/workflows/update.yml`: runs the model daily and publishes updated JSON. Requires Actions workflow permissions to write repository contents.
- `requirements.txt`: Python dependencies.

**No forecasts are fabricated.** The dashboard displays an empty-state warning until the pipeline successfully produces data. Probabilities shown by the dashboard are normal approximations based on validation residuals, not yet fully calibrated.

## Start locally
```bash
python -m pip install -r requirements.txt
python pipeline.py
python -m http.server 8000
```
Visit http://localhost:8000.

## GitHub Pages
Repository Settings → Pages → Deploy from branch → `main`, root `/`. Note that the repository is currently private and publication depends on your GitHub plan and Pages settings. Publishing the repo makes the site public. Pages hosts the static UI, not the Python backend.

## Limitations
NBA public API rate limiting or access restrictions can interrupt collection. Injury feeds, confirmed availability and projected lineups are **not** currently integrated. Predicted minutes and player forecasts are exploratory and should not be interpreted as verified or calibrated probabilities. If pipeline fails the workflow leaves the previous dataset unchanged.
