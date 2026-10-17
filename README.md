# GradeWise — Student Performance Prediction Website

A deployable website and Python API based on `student_performance_prediction.ipynb`. The user interface lets a visitor enter the six features selected in the Colab notebook, compare Linear Regression and Random Forest predictions, inspect model evaluation metrics, and run a batch prediction from a CSV file.

## Built from the supplied Colab notebook

The app mirrors the notebook's target and feature choices:

- **Target:** `G3`, the final grade on the dataset's 0–20 scale.
- **Features:** `age`, `studytime`, `failures`, `absences`, `G1`, and `G2`.
- **Models:** `LinearRegression` and `RandomForestRegressor(n_estimators=100, random_state=42)`.
- **Evaluation:** MAE, RMSE, and R², using `train_test_split(test_size=0.2, random_state=42)`.
- **Category labels:** the same thresholds as the notebook: grade ≥ 15 → High Performance; grade ≥ 10 → Average Performance; otherwise Low Performance.

The `best` model option chooses the model with the lower holdout MAE on this fixed split. This is a learning demo; it must not be used as the sole basis for consequential educational decisions.

## Recommended: export the models from Colab first

This makes deployment faster and avoids a first-request dataset download.

1. Upload `notebooks/student_performance_prediction_web_deploy.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Run all cells from top to bottom. The notebook contains the original workflow plus a final deployment-export cell.
3. The last cell downloads `student_performance_models.joblib`.
4. In this project, create/use `models/` and put the downloaded file here:

   ```text
   models/student_performance_models.joblib
   ```

5. Commit the model artifact to the same repository before deploying. It is a pickle-style trusted Python artifact: only use artifacts you created and trust. The app logs a warning if its runtime scikit-learn version differs from the version recorded in the artifact; for reproducibility, keep those versions aligned.

If you skip the export, the app tries to download the original UCI dataset and train both models on first use. That fallback requires outbound network access from the serverless function and can fail if the remote dataset is unavailable. For a more dependable deployment, commit the Colab-exported artifact.

## Run locally

Requires Python 3.11+ (Python 3.12 is specified for Vercel).

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/`. API documentation is at `http://127.0.0.1:8000/docs`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
uvicorn app:app --reload
```

## Deploy on Vercel

Vercel supports FastAPI through its Python runtime. The deployment files use Vercel's automatic framework detection; keep the static frontend in the root-level `public/` directory. On Vercel, public assets are served by the CDN and are deliberately **not** mounted from Python. `app.py` redirects `/` to `/index.html` in the deployed environment.

1. Extract this ZIP and create a new GitHub repository, for example `student-performance-prediction`.
2. Keep this exact structure at the repository root:

   ```text
   app.py
   requirements.txt
   vercel.json
   public/
     index.html
     style.css
     script.js
   services/
     __init__.py
     model_service.py
   models/
     student_performance_models.joblib  # recommended Colab export
   ```

3. Commit the files and push to GitHub:

   ```bash
   git init
   git add .
   git commit -m "Build GradeWise student performance predictor"
   git branch -M main
   git remote add origin https://github.com/YOUR-GITHUB-USERNAME/student-performance-prediction.git
   git push -u origin main
   ```

   If the repository already exists and has a remote configured, use `git add .`, `git commit ...`, and `git push` only.

4. Go to [Vercel](https://vercel.com/new), import the GitHub repository, and deploy. Set **Root Directory** to the directory containing `app.py`, `requirements.txt`, `vercel.json`, and `public/`. For this plain FastAPI/static setup, leave build and output directory overrides empty unless Vercel shows a project-specific requirement.
5. After deployment, test the endpoints below.

## Routes

| Route | Purpose |
|---|---|
| `/` | GradeWise web application (redirects to `/index.html` on Vercel) |
| `/api/health` | Lightweight API health check; does not train/load a model |
| `/api/info` | Target, selected features, and privacy/usage notes |
| `/api/predict` | Single-student grade estimate and side-by-side model predictions |
| `/api/batch-predict` | Up to 100 records in one JSON request |
| `/api/metrics` | Holdout MAE/RMSE/R² and Random Forest feature importances |
| `/docs` | Interactive FastAPI documentation |

### Single prediction request

```json
{
  "age": 17,
  "studytime": 3,
  "failures": 0,
  "absences": 5,
  "G1": 11,
  "G2": 11,
  "model": "best"
}
```

The grade prediction is bounded to the dataset's 0–20 output scale. The API is stateless: submitted values are not intentionally written to local disk or a database.

## Tests

```bash
python -m pytest -q
```

## Dataset and attribution

This project follows the notebook's use of `student-mat.csv` from the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student%2Bperformance). UCI describes it as student achievement data from two Portuguese schools and documents G1/G2 as period grades and G3 as the final-grade target. The dataset is licensed under CC BY 4.0 according to the UCI dataset page. Cite/attribute the dataset when redistributing it or publishing work based on it.

## Responsible use and limitations

- The dataset is small and specific to two Portuguese schools; accuracy can vary on students from other schools, countries, curricula, and grading systems.
- `G1` and `G2` are strong signals of `G3` in this dataset. Using prior grades can make the prediction easier but does not measure a student's potential or explain their circumstances.
- Feature importance is a model-specific ranking, not proof of causation.
- The notebook's performance metrics come from one fixed train/test split, not independent validation in your own school.
- Do not enter student names, student IDs, addresses, or other identifying information. Do not use these outputs to punish, exclude, or label students.

## Reference

Cortez, P. & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance.* Proceedings of the 5th Future Business Technology Conference. Dataset page and original citation details: UCI Machine Learning Repository.
