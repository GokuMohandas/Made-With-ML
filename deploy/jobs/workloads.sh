#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD
export RAY_AIR_REENABLE_DEPRECATED_SYNC_TO_HEAD_NODE=1
mkdir results

# Test data
export RESULTS_FILE=results/test_data_results.txt
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

# Test code
export RESULTS_FILE=results/test_code_results.txt
python -m pytest tests/code --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

# Train
export EXPERIMENT_NAME="llm"
export RESULTS_FILE=results/training_results.json
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python madewithml/train.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 1 \
    --cpu-per-worker 10 \
    --gpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp $RESULTS_FILE

# Get and save run ID
export RUN_ID=$(python -c "import os; from madewithml import utils; d = utils.load_dict(os.getenv('RESULTS_FILE')); print(d['run_id'])")
export RUN_ID_FILE=results/run_id.txt
echo $RUN_ID > $RUN_ID_FILE  # used for serving later

# Evaluate
export RESULTS_FILE=results/evaluation_results.json
export HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"
python madewithml/evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc $HOLDOUT_LOC \
    --results-fp $RESULTS_FILE

# Test model
RESULTS_FILE=results/test_model_results.txt
pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

# Save to S3
export MODEL_REGISTRY=$(python -c "from madewithml import config; print(config.MODEL_REGISTRY)")
aws s3 cp $MODEL_REGISTRY s3://madewithml/$GITHUB_USERNAME/mlflow/ --recursive
aws s3 cp results/ s3://madewithml/$GITHUB_USERNAME/results/ --recursive
