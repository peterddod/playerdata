## Player Tracking Data — Setup and Usage

This repository contains a small toolkit for cleaning, filtering, analyzing, and visualizing player tracking data, along with an example notebook.

- Code lives in `lib/`
- Example analysis in `analysis.ipynb`
- Put your input files in `data/`
- Generated figures can be saved to `figures/`

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

