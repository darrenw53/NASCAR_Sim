# NASCAR Simulator + FanDuel Optimizer (Streamlit)

This is a starter, end-to-end NASCAR Monte Carlo simulator with slider-controlled weighting of:
- Past track results
- Practice
- Qualifying
- Season baseline strength

It then blends **simulated FanDuel points** with FanDuel's slate **FPPG** and runs a simple optimizer.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Weekly data folders

Add one folder per race under:

`data/weekly/<race_slug>/`

Recommended layout:

```
data/weekly/daytona_500/
  notes.json
  practice.json
  qualifying.json
  race_results/
    2025.json
    2024.json
    2023.json
```

### notes.json

Minimum:

```json
{
  "track_label": "Daytona International Speedway",
  "track_type": "superspeedway",
  "total_laps": 200
}
```

## FanDuel slate file

Place the weekly export under:

`data/fan_duel/week_players.csv`

and keep the standard FanDuel columns: First Name, Last Name, Salary, FPPG.

## Outputs

When you click **Run simulation**, CSVs are written to:

- `outputs/projections/<race_slug>_projections.csv`
- `outputs/sims/<race_slug>_sims_long.csv`
