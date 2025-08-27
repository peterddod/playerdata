## Player Data Task

This repository contains the solution to the takehome task for the Player Data R&D role.

- Code lives in `lib/`
- Analysis in `analysis.ipynb`
- Raw file in `data/`
- Generated figures saved to `figures/`

### Prerequisites
- Python 3.10 or newer

### Setup (Windows PowerShell)
```powershell
cd C:\repos\playerdata
python -m venv .venv
.\.venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
```

To leave the environment later:
```powershell
deactivate
```

## Data expectations
If there is no data in the `/data` folder, add the `match_data.csv` file to the folder. It must have this name for the `analysis.ipynb` to run.


### Open the notebook
```powershell
jupyter lab  # or: jupyter notebook
```
Then open `analysis.ipynb` and run all the cells.

---

