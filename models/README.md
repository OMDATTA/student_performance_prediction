# Exported model artifact

Run the final export cell in `notebooks/student_performance_prediction_web_deploy.ipynb`,
then put `student_performance_models.joblib` in this directory. Commit that artifact
for faster, deterministic inference on Vercel. Do not load model pickle/joblib files
from sources you do not trust.

If the artifact is absent, the API attempts to download the public UCI dataset and
train the two models on first use. That fallback depends on outbound network access.
