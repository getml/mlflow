"""
The ``mlflow.sklearn`` module provides an API for logging and loading scikit-learn models. This
module exports scikit-learn models with the following flavors:

Python (native) `pickle <https://scikit-learn.org/stable/modules/model_persistence.html>`_ format
    This is the main flavor that can be loaded back into scikit-learn.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
    NOTE: The `mlflow.pyfunc` flavor is only added for scikit-learn models that define `predict()`,
    since `predict()` is required for pyfunc model inference.
"""

import functools
import inspect
import logging
import os
import pickle
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _inspect_original_var_name, gorilla
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
    MlflowAutologgingQueueingClient,
    _get_new_training_session_class,
    autologging_integration,
    disable_autologging,
    get_autologging_config,
    get_instance_method_first_arg_value,
    resolve_input_example_and_signature,
    safe_patch,
    update_wrapper_extended,
)
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
    MLFLOW_AUTOLOGGING,
    MLFLOW_DATASET_CONTEXT,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "getml"

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [SERIALIZATION_FORMAT_PICKLE, SERIALIZATION_FORMAT_CLOUDPICKLE]

_logger = logging.getLogger(__name__)
_SklearnTrainingSession = _get_new_training_session_class()

_MODEL_DATA_SUBPATH = "model.pkl"


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="getml"))
def save_model(
    getml_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn="predict",
    metadata=None,
):
    """
    Save a scikit-learn model to a path on the local file system. Produces a MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.sklearn`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for scikit-learn models
          that define `predict()`, since `predict()` is required for pyfunc model inference.

    Args:
        sk_model: scikit-learn model to be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        serialization_format: The format in which to serialize the model. This should be one of
            the formats listed in
            ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
            format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
            provides better cross-system compatibility by identifying and
            packaging code dependencies with the serialized model.

        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        pyfunc_predict_fn: The name of the prediction function to use for inference with the
            pyfunc representation of the resulting MLflow Model. Current supported functions
            are: ``"predict"``, ``"predict_proba"``, ``"predict_log_proba"``,
            ``"predict_joint_log_proba"``, and ``"score"``.
        metadata: {{ metadata }}

    .. code-block:: python
        :caption: Example

        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn import tree

        iris = load_iris()
        sk_model = tree.DecisionTreeClassifier()
        sk_model = sk_model.fit(iris.data, iris.target)

        # Save the model in cloudpickle format
        # set path to location for persistence
        sk_path_dir_1 = ...
        mlflow.sklearn.save_model(
            sk_model,
            sk_path_dir_1,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # save the model in pickle format
        # set path to location for persistence
        sk_path_dir_2 = ...
        mlflow.sklearn.save_model(
            sk_model,
            sk_path_dir_2,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
    """
    import getml

    #_validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                f"Unrecognized serialization format: {serialization_format}. Please specify one"
                f" of the following supported formats: {SUPPORTED_SERIALIZATION_FORMATS}."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    _validate_and_prepare_target_save_path(path)
    code_path_subdir = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    # saved_example = _save_example(mlflow_model, input_example, path)

    # if signature is None and saved_example is not None:
    #     wrapped_model = _SklearnModelWrapper(sk_model)
    #     signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    # elif signature is False:
    #     signature = None

    # if signature is not None:
    #     mlflow_model.signature = signature
    # if metadata is not None:
    #     mlflow_model.metadata = metadata

    model_data_subpath = _MODEL_DATA_SUBPATH
    model_data_path = os.path.join(path, model_data_subpath)
    _save_model(
        getml_model=getml_model,
        output_path=model_data_path,
        serialization_format=serialization_format,
    )

    # `PyFuncModel` only works for sklearn models that define a predict function

    if hasattr(getml_model, pyfunc_predict_fn):
        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.getml",
            model_path=model_data_subpath,
            conda_env=_CONDA_ENV_FILE_NAME,
            python_env=_PYTHON_ENV_FILE_NAME,
            code=code_path_subdir,
            predict_fn=pyfunc_predict_fn,
        )
    else:
        _logger.warning(
            f"Model was missing function: {pyfunc_predict_fn}. Not logging python_function flavor!"
        )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        getml_version=getml.__version__,
        serialization_format=serialization_format,
        code=code_path_subdir,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    # if conda_env is None:
    #     if pip_requirements is None:
    #         include_cloudpickle = serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE
    #         default_reqs = get_default_pip_requirements(include_cloudpickle)
    #         # To ensure `_load_pyfunc` can successfully load the model during the dependency
    #         # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
    #         inferred_reqs = mlflow.models.infer_pip_requirements(
    #             model_data_path,
    #             FLAVOR_NAME,
    #             fallback=default_reqs,
    #         )
    #         default_reqs = sorted(set(inferred_reqs).union(default_reqs))
    #     else:
    #         default_reqs = None
    #     conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
    #         default_reqs,
    #         pip_requirements,
    #         extra_pip_requirements,
    #     )
    # else:
    #     conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    # with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
    #     yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # # Save `constraints.txt` if necessary
    # if pip_constraints:
    #     write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # # Save `requirements.txt`
    # write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    # _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))

def _save_model(getml_model, output_path, serialization_format):
    """
    Args:
        sk_model: The scikit-learn model to serialize.
        output_path: The file path to which to write the serialized model.
        serialization_format: The format in which to serialize the model. This should be one of
            the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
            ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    with open(output_path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            _dump_model(pickle, getml_model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            _dump_model(cloudpickle, getml_model, out)
        else:
            raise MlflowException(
                message=f"Unrecognized serialization format: {serialization_format}",
                error_code=INTERNAL_ERROR,
            )

def _dump_model(pickle_lib, getml_model, out):
    try:
        # Using python's default protocol to optimize compatibility.
        # Otherwise cloudpickle uses latest protocol leading to incompatibilities.
        # See https://github.com/mlflow/mlflow/issues/5419
        pickle_lib.dump(getml_model, out, protocol=pickle.DEFAULT_PROTOCOL)
    except:
        pass
    # except (pickle.PicklingError, TypeError, AttributeError) as e:
    #     if sk_model.__class__ not in _gen_estimators_to_patch():
    #         raise _SklearnCustomModelPicklingError(sk_model, e)
    #     else:
    #         raise

@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="getml"))
def log_model(
    getml_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn="predict",
    metadata=None,
):
    """
    Log a scikit-learn model as an MLflow artifact for the current run. Produces an MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.sklearn`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for scikit-learn models
          that define `predict()`, since `predict()` is required for pyfunc model inference.

    Args:
        sk_model: scikit-learn model to be saved.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        serialization_format: The format in which to serialize the model. This should be one of
            the formats listed in
            ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
            format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
            provides better cross-system compatibility by identifying and
            packaging code dependencies with the serialized model.
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.
        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        pyfunc_predict_fn: The name of the prediction function to use for inference with the
            pyfunc representation of the resulting MLflow Model. Current supported functions
            are: ``"predict"``, ``"predict_proba"``, ``"predict_log_proba"``,
            ``"predict_joint_log_proba"``, and ``"score"``.
        metadata: {{ metadata }}

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature
        from sklearn.datasets import load_iris
        from sklearn import tree

        with mlflow.start_run():
            # load dataset and train model
            iris = load_iris()
            sk_model = tree.DecisionTreeClassifier()
            sk_model = sk_model.fit(iris.data, iris.target)

            # log model params
            mlflow.log_param("criterion", sk_model.criterion)
            mlflow.log_param("splitter", sk_model.splitter)
            signature = infer_signature(iris.data, sk_model.predict(iris.data))

            # log model
            mlflow.sklearn.log_model(sk_model, "sk_models", signature=signature)

    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.getml,
        getml_model=getml_model,
        conda_env=conda_env,
        code_paths=code_paths,
        serialization_format=serialization_format,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        pyfunc_predict_fn=pyfunc_predict_fn,
        metadata=metadata,
    )

def load_model(model_uri, dst_path=None):
    """
    Load a scikit-learn model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model, for example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        A scikit-learn model.

    .. code-block:: python
        :caption: Example

        import mlflow.sklearn

        sk_model = mlflow.sklearn.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2/sk_models")

        # use Pandas DataFrame to make predictions
        pandas_df = ...
        predictions = sk_model.predict(pandas_df)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    sklearn_model_artifacts_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    serialization_format = flavor_conf.get("serialization_format", SERIALIZATION_FORMAT_PICKLE)
    return _load_model_from_local_file(
        path=sklearn_model_artifacts_path, serialization_format=serialization_format
    )

def _load_model_from_local_file(path, serialization_format):
    """Load a scikit-learn model saved as an MLflow artifact on the local file system.

    Args:
        path: Local filesystem path to the MLflow Model saved with the ``sklearn`` flavor
        serialization_format: The format in which the model was serialized. This should be one of
            the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
            ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    # TODO: we could validate the scikit-learn version here
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                f"Unrecognized serialization format: {serialization_format}. Please specify one"
                f" of the following supported formats: {SUPPORTED_SERIALIZATION_FORMATS}."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )
    with open(path, "rb") as f:
        # Models serialized with Cloudpickle cannot necessarily be deserialized using Pickle;
        # That's why we check the serialization format of the model before deserializing
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(f)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            return cloudpickle.load(f)




# class GetMLWrapper(PythonModel):
#     def __init__(self):
#         self.model = None

#     def load_context(self, context):
#         from joblib import load

#         self.model = load(context.artifacts["model_path"])

#     def predict(self, context, model_input, params=None):

#         container_unseen = getml.data.Container(model_input['population'])
#         container_unseen.add(peripheral=model_input['peripheral'])

#         return self.model.predict(container_unseen.full)