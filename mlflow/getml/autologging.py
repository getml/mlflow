import json
from dataclasses import dataclass, field
from typing import Any
import threading

import mlflow
from mlflow.utils import gorilla
from mlflow.utils.autologging_utils import safe_patch
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient


@dataclass
class LogInfo:
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)


def autolog(
    flavor_name,
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    max_tuning_runs=5,
    log_post_training_metrics=True,
):
    flavor_name = "getml"
    import getml
    from dataclasses import fields, dataclass, is_dataclass

    def _patch_pipeline_method(flavor_name, class_def, func_name, patched_fn, manage_run):
        print(f"patching {flavor_name}.{class_def.__name__}.{func_name}")
        safe_patch(
            flavor_name,
            class_def,
            func_name,
            patched_fn,
            manage_run=manage_run,
        )
        print(f"done patching {flavor_name}.{class_def.__name__}.{func_name}")

    def _extract_pipeline_informations(getml_pipeline: getml.Pipeline) -> LogInfo:
        params = (
            "preprocessors",
            "feature_learners",
            "feature_selectors",
            "predictors",
            "loss_function",
        )
        pipeline_informations = {}

        for parameter_name in params:
            values = getattr(getml_pipeline, parameter_name)
            if isinstance(values, list):
                for v in values:
                    if is_dataclass(v):
                        name = v.__class__.__name__
                        for field in fields(v):
                            field_value = getattr(v, field.name)
                            if isinstance(field_value, (frozenset, set)):
                                field_value = json.dumps(list(field_value))
                            elif not isinstance(field_value, str):
                                field_value = json.dumps(field_value)
                            pipeline_informations[f"{parameter_name}.{name}.{field.name}"] = (
                                field_value
                            )
            # else:
            #    value_name = values.__class__.__name__
            #    pipeline_informations[parameter_name] = value_name
        tags = [str(t) for t in getml_pipeline.tags]
        return LogInfo(params=pipeline_informations, tags=dict(zip(tags, tags)))

    def _extract_fitted_pipeline_informations(getml_pipeline: getml.Pipeline) -> LogInfo:
        params = {
            "pipeline_id": getml_pipeline.id,
        }

        metrics = {}

        scores = getml_pipeline.scores

        if getml_pipeline.is_classification:
            metrics["auc"] = scores.auc
            metrics["accuracy"] = scores.accuracy
            metrics["cross_entropy"] = scores.cross_entropy

        if getml_pipeline.is_regression:
            metrics["mae"] = scores.mae
            metrics["rmse"] = scores.rmse
            metrics["rsquared"] = scores.rsquared

        # for feature in getml_pipeline.features:
        #     metrics[f"{feature.name}.importance"] = json.dumps(feature.importance)
        #     metrics[f"{feature.name}.correlation"] = json.dumps(feature.correlation)

        # if len(getml_pipeline.targets) == 1:
        #     metrics["targets"] = getml_pipeline.targets[0]
        # else:
        #     for i, t in enumerate(getml_pipeline.targets):
        #         metrics[f"targets.{i}"] = t
        return LogInfo(
            params=params,
            metrics=metrics,
        )

    def _extract_engine_system_metrics(autologging_client, run_id, stop_event):
        import requests
        import numpy as np

        memory_usage_url = "http://localhost:1709/getmemoryusage/"
        cpu_usage_url = "http://localhost:1709/getcpuusage/"

        step = 0
        while not stop_event.is_set():
            memory_usage_response = requests.get(memory_usage_url)
            cpu_usage_response = requests.get(cpu_usage_url)
            memory_usage_data = memory_usage_response.json()["data"][0][-1]
            cpu_usage_data = cpu_usage_response.json()["data"][0][-1]
            autologging_client.log_metrics(
                run_id=run_id,
                metrics={
                    "engine_cpu_usage_per_virtual_core_in_pct": np.round(cpu_usage_data, 2),
                    "memory_usage_in_pct": np.round(memory_usage_data, 2),
                },
                step=step,
            )
            step += 1
            stop_event.wait(1)

    def patched_fit_mlflow(original, self: getml.Pipeline, *args, **kwargs):
        autologging_client = MlflowAutologgingQueueingClient()
        assert (active_run := mlflow.active_run())
        run_id = active_run.info.run_id
        pipeline_log_info = _extract_pipeline_informations(self)
        autologging_client.log_params(
            run_id=run_id,
            params=pipeline_log_info.params,
        )
        if tags := pipeline_log_info.tags:
            autologging_client.set_tags(run_id=run_id, tags=tags)

        stop_event = threading.Event()
        metrics_thread = threading.Thread(
            target=_extract_engine_system_metrics, args=(autologging_client, run_id, stop_event)
        )
        metrics_thread.start()

        fit_output = original(self, *args, **kwargs)

        stop_event.set()
        metrics_thread.join()

        fitted_pipeline_log_info = _extract_fitted_pipeline_informations(self)
        autologging_client.log_metrics(
            run_id=run_id,
            metrics=fitted_pipeline_log_info.metrics,
        )

        autologging_client.flush(synchronous=True)
        return fit_output

    _patch_pipeline_method(
        flavor_name=flavor_name,
        class_def=getml.pipeline.Pipeline,
        func_name="fit",
        patched_fn=patched_fit_mlflow,
        manage_run=True,
    )
