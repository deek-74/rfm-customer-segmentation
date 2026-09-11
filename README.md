# RFM Customer Segmentation

Excel workbook and HTML dashboard for RFM customer segmentation on the [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) extract (UK giftware, Dec 2010 – 9 Dec 2011).

**Final deliverables**

| File | What it is |
|---|---|
| `outputs/RFM_Customer_Segmentation_Workbook.xlsx` | The workbook to present |
| `outputs/RFM_Executive_Dashboard.html` | Browser dashboard |

**4,324 customers · £8.25m historical net · Champions are 18% of customers and 63% of revenue**

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Save the UCI file as `data/raw/Online Retail.xlsx` (space in the name), then:

```powershell
python run.py all
```

That builds the workbook and dashboard in `outputs/`.

| Command | What it does |
|---|---|
| `python run.py data` | Excel → parquet |
| `python run.py pipeline` | Clean, RFM, segments, clusters |
| `python run.py excel` | Write the workbook and HTML dashboard |

The `.xlsx` is ~50 MB. Do not commit it; rebuild with `python run.py excel`.
