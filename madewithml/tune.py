import datetime
import json
import os

import ray
import typer
from ray import tune
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from typing_extensions import Annotated

from madewithml import data, train, utils
from madewithml.config import EFS_DIR, MLFLOW_TRACKING_URI, logger

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def tune_models(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment for this training workload.")] = None,
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset.")] = None,
    initial_params: Annotated[str, typer.Option(help="initial config for the tuning workload.")] = None,
    num_workers: Annotated[int, typer.Option(help="number of workers to use for training.")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="number of CPUs to use per worker.")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="number of GPUs to use per worker.")] = 0,
    num_runs: Annotated[int, typer.Option(help="number of runs in this tuning experiment.")] = 1,
    num_samples: Annotated[int, typer.Option(help="number of samples to use from dataset.")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for.")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch.")] = 256,
    results_fp: Annotated[str, typer.Option(help="filepath to save results to.")] = None,
) -> ray.tune.result_grid.ResultGrid:
    """Hyperparameter tuning experiment.

    Args:
        experiment_name (str): name of the experiment for this training workload.
        dataset_loc (str): location of the dataset.
        initial_params (str): initial config for the tuning workload.
        num_workers (int, optional): number of workers to use for training. Defaults to 1.
        cpu_per_worker (int, optional): number of CPUs to use per worker. Defaults to 1.
        gpu_per_worker (int, optional): number of GPUs to use per worker. Defaults to 0.
        num_runs (int, optional): number of runs in this tuning experiment. Defaults to 1.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.
        results_fp (str, optional): filepath to save the tuning results. Defaults to None.

    Returns:
        ray.tune.result_grid.ResultGrid: results of the tuning experiment.
    """
    # Set up
    utils.set_seeds()
    train_loop_config = {}
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
    )

    # Dataset
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config.get("num_samples", None))
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    tags = train_ds.unique(column="tag")
    train_loop_config["num_classes"] = len(tags)

    # Dataset config
    dataset_config = {
        "train": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
        "val": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
    }

    # Preprocess
    preprocessor = data.CustomPreprocessor()
    preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train.train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        metadata={"class_to_index": preprocessor.class_to_index},
    )

    # Checkpoint configuration
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # Run configuration
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True,
    )
    run_config = RunConfig(callbacks=[mlflow_callback], checkpoint_config=checkpoint_config, storage_path=EFS_DIR, local_dir=EFS_DIR)

    # Hyperparameters to start with
    initial_params = json.loads(initial_params)
    search_alg = HyperOptSearch(points_to_evaluate=initial_params)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)  # trade off b/w optimization and search space

    # Parameter space
    param_space = {
        "train_loop_config": {
            "dropout_p": tune.uniform(0.3, 0.9),
            "lr": tune.loguniform(1e-5, 5e-4),
            "lr_factor": tune.uniform(0.1, 0.9),
            "lr_patience": tune.uniform(1, 10),
        }
    }

    # Scheduler
    scheduler = AsyncHyperBandScheduler(
        max_t=train_loop_config["num_epochs"],  # max epoch (<time_attr>) per trial
        grace_period=1,  # min epoch (<time_attr>) per trial
    )

    # Tune config
    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=num_runs,
    )

    # Tuner
    tuner = Tuner(
        trainable=trainer,
        run_config=run_config,
        param_space=param_space,
        tune_config=tune_config,
    )

    # Tune
    results = tuner.fit()
    best_trial = results.get_best_result(metric="val_loss", mode="min")
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(experiment_name=experiment_name, trial_id=best_trial.metrics["trial_id"]),
        "params": best_trial.config["train_loop_config"],
        "metrics": utils.dict_to_list(best_trial.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]),
    }
    logger.info(json.dumps(d, indent=2))
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(d, results_fp)
    return results


if __name__ == "__main__":  # pragma: no cover, application
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"env_vars": {"GITHUB_USERNAME": os.environ["GITHUB_USERNAME"]}})
    app()
