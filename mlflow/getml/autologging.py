import json
from dataclasses import dataclass, field
from typing import Any

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
                #original = gorilla.get_original_attribute(
                #    class_def,
                #    func_name,
                #    bypass_descriptor_protocol=False
                #)
        safe_patch(
            flavor_name,
            class_def,
            func_name,
            patched_fn,
            manage_run=manage_run,
        )

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
                            if not isinstance(field_value, str):
                                field_value = json.dumps(field_value)
                            pipeline_informations[f"{parameter_name}.{name}.{field.name}"] = field_value
            #else:
            #    value_name = values.__class__.__name__
            #    pipeline_informations[parameter_name] = value_name
        tags = [str(t) for t in getml_pipeline.tags ]
        return LogInfo(
            params=pipeline_informations,
            tags=dict(zip(tags, tags))
        )

    def _extract_fitted_pipeline_informations(getml_pipeline: getml.Pipeline) -> LogInfo:
        params ={
            "pipeline_id": getml_pipeline.id,
        }
        metrics = {
        }
        if len(getml_pipeline.targets) == 1:
            params["targets"] = getml_pipeline.targets[0]
        else:
            for i, t in enumerate(getml_pipeline.targets):
                params[f"targets.{i}"] = t
        return LogInfo(
            params=params,
            metrics=metrics,
        ) 

    def _get_X_y(fit_func, fit_args, fit_kwargs):
        population = fit_kwargs.get("population")
        peripheral = fit_kwargs.get("peripheral")


    def patched_fit_mlflow(original, self: getml.Pipeline, *args, **kwargs):
        autologging_client = MlflowAutologgingQueueingClient()
        assert (active_run := mlflow.active_run())
        run_id = active_run.info.run_id
        pipeline_log_info = _extract_pipeline_informations(self)
        autologging_client.log_params(
            run_id = run_id,
            params=pipeline_log_info.params, 
        )
        if (tags := pipeline_log_info.tags):
            autologging_client.set_tags(run_id=run_id, tags=tags)

        # CALL
        fit_output = original(self, *args, **kwargs)
        autologging_client.flush(synchronous=True)
        return fit_output


    _patch_pipeline_method(
        flavor_name=flavor_name,
        class_def=getml.pipeline.Pipeline,
        func_name="fit",
        patched_fn=patched_fit_mlflow,
        manage_run=True, 
    ) 




    
