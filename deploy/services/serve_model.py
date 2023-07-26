import os
import subprocess
import sys

sys.path.append(".")

from madewithml.config import MODEL_REGISTRY  # NOQA: E402
from madewithml.serve import ModelDeployment  # NOQA: E402

# Copy from S3
github_username = os.environ.get("GITHUB_USERNAME")
subprocess.check_output(["aws", "s3", "cp", f"s3://madewithml/{github_username}/mlflow/", str(MODEL_REGISTRY), "--recursive"])
subprocess.check_output(["aws", "s3", "cp", f"s3://madewithml/{github_username}/results/", "./", "--recursive"])

# Entrypoint
run_id = [line.strip() for line in open("run_id.txt")][0]
entrypoint = ModelDeployment.bind(run_id=run_id, threshold=0.9)
