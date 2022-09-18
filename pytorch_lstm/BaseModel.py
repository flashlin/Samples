import matplotlib.pyplot as plt
import warnings
import numpy as np
from numpy.lib.function_base import iterable
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import copy
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.nn.utils import rnn
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, GroupNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.optim import Ranger
from pytorch_forecasting.utils import (
    InitialParameterRepresenterMixIn,
    OutputMixIn,
    TupleOutputMixIn,
    apply_to_list,
    create_mask,
    get_embedding_size,
    groupby_apply,
    move_to_device,
    to_list,
)
from pytorch_forecasting.metrics import (
    MAE,
    MASE,
    SMAPE,
    DistributionLoss,
    MultiHorizonMetric,
    MultiLoss,
    QuantileLoss,
    convert_torchmetric_to_pytorch_forecasting_metric,
)
from pytorch_lightning.utilities.parsing import AttributeDict, get_init_args
from pytorch_forecasting.metrics.base_metrics import Metric
import inspect
import yaml

def _torch_cat_na(x: List[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate tensor along ``dim=0`` and add nans along ``dim=1`` if necessary.

    Allows concatenation of tensors where ``dim=1`` are not equal.
    Missing values are filled up with ``nan``.

    Args:
        x (List[torch.Tensor]): list of tensors to concatenate along dimension 0

    Returns:
        torch.Tensor: concatenated tensor
    """
    if x[0].ndim > 1:
        first_lens = [xi.shape[1] for xi in x]
        max_first_len = max(first_lens)
        if max_first_len > min(first_lens):
            x = [
                xi
                if xi.shape[1] == max_first_len
                else torch.cat(
                    [xi, torch.full((xi.shape[0], max_first_len - xi.shape[1], *xi.shape[2:]), float("nan"))], dim=1
                )
                for xi in x
            ]

    # check if remaining dimensions are all equal
    if x[0].ndim > 2:
        remaining_dimensions_equal = all([all([xi.size(i) == x[0].size(i) for xi in x]) for i in range(2, x[0].ndim)])
    else:
        remaining_dimensions_equal = True

    # deaggregate
    if remaining_dimensions_equal:
        return torch.cat(x, dim=0)
    else:
        # make list instead but warn
        warnings.warn(
            f"Not all dimensions are equal for tensors shapes. Example tensor {x[0].shape}. "
            "Returning list instead of torch.Tensor.",
            UserWarning,
        )
        return [xii for xi in x for xii in xi]



def _concatenate_output(
    output: List[Dict[str, List[Union[List[torch.Tensor], torch.Tensor, bool, int, str, np.ndarray]]]]
) -> Dict[str, Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, int, bool, str]]]]:
    """
    Concatenate multiple batches of output dictionary.

    Args:
        output (List[Dict[str, List[Union[List[torch.Tensor], torch.Tensor, bool, int, str, np.ndarray]]]]):
            list of outputs to concatenate. Each entry corresponds to a batch.

    Returns:
        Dict[str, Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, int, bool, str]]]]:
            concatenated output
    """
    output_cat = {}
    for name in output[0].keys():
        v0 = output[0][name]
        # concatenate simple tensors
        if isinstance(v0, torch.Tensor):
            output_cat[name] = _torch_cat_na([out[name] for out in output])
        # concatenate list of tensors
        elif isinstance(v0, (tuple, list)) and len(v0) > 0:
            output_cat[name] = []
            for target_id in range(len(v0)):
                if isinstance(v0[target_id], torch.Tensor):
                    output_cat[name].append(_torch_cat_na([out[name][target_id] for out in output]))
                else:
                    try:
                        output_cat[name].append(np.concatenate([out[name][target_id] for out in output], axis=0))
                    except ValueError:
                        output_cat[name] = [item for out in output for item in out[name][target_id]]
        # flatten list for everything else
        else:
            try:
                output_cat[name] = np.concatenate([out[name] for out in output], axis=0)
            except ValueError:
                if iterable(output[0][name]):
                    output_cat[name] = [item for out in output for item in out[name]]
                else:
                    output_cat[name] = [out[name] for out in output]

    if isinstance(output[0], OutputMixIn):
        output_cat = output[0].__class__(**output_cat)
    return output_cat

STAGE_STATES = {
    RunningStage.TRAINING: "train",
    RunningStage.VALIDATING: "val",
    RunningStage.TESTING: "test",
    RunningStage.PREDICTING: "predict",
    RunningStage.SANITY_CHECKING: "sanity_check",
}

class BaseModel(InitialParameterRepresenterMixIn, LightningModule, TupleOutputMixIn):
    """
    BaseModel from which new timeseries models should inherit from.
    The ``hparams`` of the created object will default to the parameters indicated in :py:meth:`~__init__`.

    The :py:meth:`~BaseModel.forward` method should return a named tuple with at least the entry ``prediction``
    that contains the network's output. See the function's documentation for more details.

    The idea of the base model is that common methods do not have to be re-implemented for every new architecture.
    The class is a [LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html)
    and follows its conventions. However, there are important additions:

        * You need to specify a ``loss`` attribute that stores the function to calculate the
          :py:class:`~pytorch_forecasting.metrics.MultiHorizonLoss` for backpropagation.
        * The :py:meth:`~BaseModel.from_dataset` method can be used to initialize a network using the specifications
          of a dataset. Often, parameters such as the number of features can be easily deduced from the dataset.
          Further, the method will also store how to rescale normalized predictions into the unnormalized prediction
          space. Override it to pass additional arguments to the __init__ method of your network that depend on your
          dataset.
        * The :py:meth:`~BaseModel.transform_output` method rescales the network output using the target normalizer
          from thedataset.
        * The :py:meth:`~BaseModel.step` method takes care of calculating the loss, logging additional metrics defined
          in the ``logging_metrics`` attribute and plots of sample predictions. You can override this method to add
          custom interpretations or pass extra arguments to the networks forward method.
        * The :py:meth:`~BaseModel.epoch_end` method can be used to calculate summaries of each epoch such as
          statistics on the encoder length, etc.
        * The :py:meth:`~BaseModel.predict` method makes predictions using a dataloader or dataset. Override it if you
          need to pass additional arguments to ``forward`` by default.

    To implement your own architecture, it is best to
    go throught the :ref:`Using custom data and implementing custom models <new-model-tutorial>` and
    to look at existing ones to understand what might be a good approach.

    Example:

        .. code-block:: python

            class Network(BaseModel):

                def __init__(self, my_first_parameter: int=2, loss=SMAPE()):
                    self.save_hyperparameters()
                    super().__init__(loss=loss)

                def forward(self, x):
                    normalized_prediction = self.module(x)
                    prediction = self.transform_output(prediction=normalized_prediction, target_scale=x["target_scale"])
                    return self.to_network_output(prediction=prediction)

    """

    CHECKPOINT_HYPER_PARAMS_SPECIAL_KEY = "__special_save__"

    def __init__(
        self,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        learning_rate: Union[float, List[float]] = 1e-3,
        log_gradient_flow: bool = False,
        loss: Metric = SMAPE(),
        logging_metrics: nn.ModuleList = nn.ModuleList([]),
        reduce_on_plateau_patience: int = 1000,
        reduce_on_plateau_reduction: float = 2.0,
        reduce_on_plateau_min_lr: float = 1e-5,
        weight_decay: float = 0.0,
        optimizer_params: Dict[str, Any] = None,
        monotone_constaints: Dict[str, int] = {},
        output_transformer: Callable = None,
        optimizer="ranger",
    ):
        """
        BaseModel for timeseries forecasting from which to inherit from

        Args:
            log_interval (Union[int, float], optional): Batches after which predictions are logged. If < 1.0, will log
                multiple entries per batch. Defaults to -1.
            log_val_interval (Union[int, float], optional): batches after which predictions for validation are
                logged. Defaults to None/log_interval.
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            log_gradient_flow (bool): If to log gradient flow, this takes time and should be only done to diagnose
                training failures. Defaults to False.
            loss (Metric, optional): metric to optimize, can also be list of metrics. Defaults to SMAPE().
            logging_metrics (nn.ModuleList[MultiHorizonMetric]): list of metrics that are logged during training.
                Defaults to [].
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10. Defaults
                to 1000
            reduce_on_plateau_reduction (float): reduction in learning rate when encountering plateau. Defaults to 2.0.
            reduce_on_plateau_min_lr (float): minimum learning rate for reduce on plateua learning rate scheduler.
                Defaults to 1e-5
            weight_decay (float): weight decay. Defaults to 0.0.
            optimizer_params (Dict[str, Any]): additional parameters for the optimizer. Defaults to {}.
            monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
                variables mapping
                position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
                larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
                This constraint significantly slows down training. Defaults to {}.
            output_transformer (Callable): transformer that takes network output and transforms it to prediction space.
                Defaults to None which is equivalent to ``lambda out: out["prediction"]``.
            optimizer (str): Optimizer, "ranger", "sgd", "adam", "adamw" or class name of optimizer in ``torch.optim``.
                Alternatively, a class or function can be passed which takes parameters as first argument and
                a `lr` argument (optionally also `weight_decay`)
                Defaults to "ranger".
        """
        super().__init__()
        # update hparams
        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        self.save_hyperparameters(
            {name: val for name, val in init_args.items() if name not in self.hparams and name not in ["self"]}
        )

        # update log interval if not defined
        if self.hparams.log_val_interval is None:
            self.hparams.log_val_interval = self.hparams.log_interval

        if not hasattr(self, "loss"):
            if isinstance(loss, (tuple, list)):
                self.loss = MultiLoss(metrics=[convert_torchmetric_to_pytorch_forecasting_metric(l) for l in loss])
            else:
                self.loss = convert_torchmetric_to_pytorch_forecasting_metric(loss)
        if not hasattr(self, "logging_metrics"):
            self.logging_metrics = nn.ModuleList(
                [convert_torchmetric_to_pytorch_forecasting_metric(l) for l in logging_metrics]
            )
        if not hasattr(self, "output_transformer"):
            self.output_transformer = output_transformer
        if not hasattr(self, "optimizer"):  # callables are removed from hyperparameters, so better to save them
            self.optimizer = self.hparams.optimizer

        # delete everything from hparams that cannot be serialized with yaml.dump
        # which is particularly important for tensorboard logging
        hparams_to_delete = []
        for k, v in self.hparams.items():
            try:
                yaml.dump(v)
            except:  # noqa
                hparams_to_delete.append(k)
                if not hasattr(self, k):
                    setattr(self, k, v)

        self.hparams_special = getattr(self, "hparams_special", [])
        self.hparams_special.extend(hparams_to_delete)
        for k in hparams_to_delete:
            del self._hparams[k]
            del self._hparams_initial[k]


    @property
    def current_stage(self) -> str:
        """
        Available inside lightning loops.
        :return: current trainer stage. One of ["train", "val", "test", "predict", "sanity_check"]
        """
        return STAGE_STATES[self.trainer.state.stage]

    @property
    def n_targets(self) -> int:
        """
        Number of targets to forecast.

        Based on loss function.

        Returns:
            int: number of targets
        """
        if isinstance(self.loss, MultiLoss):
            return len(self.loss.metrics)
        else:
            return 1

    def transform_output(
        self,
        prediction: Union[torch.Tensor, List[torch.Tensor]],
        target_scale: Union[torch.Tensor, List[torch.Tensor]],
        loss: Optional[Metric] = None,
    ) -> torch.Tensor:
        """
        Extract prediction from network output and rescale it to real space / de-normalize it.

        Args:
            prediction (Union[torch.Tensor, List[torch.Tensor]]): normalized prediction
            target_scale (Union[torch.Tensor, List[torch.Tensor]]): scale to rescale prediction
            loss (Optional[Metric]): metric to use for transform

        Returns:
            torch.Tensor: rescaled prediction
        """
        if loss is None:
            loss = self.loss
        if isinstance(loss, MultiLoss):
            out = loss.rescale_parameters(
                prediction,
                target_scale=target_scale,
                encoder=self.output_transformer.normalizers,  # need to use normalizer per encoder
            )
        else:
            out = loss.rescale_parameters(prediction, target_scale=target_scale, encoder=self.output_transformer)
        return out


    @staticmethod
    def deduce_default_output_parameters(
        dataset: TimeSeriesDataSet, kwargs: Dict[str, Any], default_loss: MultiHorizonMetric = None
    ) -> Dict[str, Any]:
        """
        Deduce default parameters for output for `from_dataset()` method.

        Determines ``output_size`` and ``loss`` parameters.

        Args:
            dataset (TimeSeriesDataSet): timeseries dataset
            kwargs (Dict[str, Any]): current hyperparameters
            default_loss (MultiHorizonMetric, optional): default loss function.
                Defaults to :py:class:`~pytorch_forecasting.metrics.MAE`.

        Returns:
            Dict[str, Any]: dictionary with ``output_size`` and ``loss``.
        """
        # infer output size
        def get_output_size(normalizer, loss):
            if isinstance(loss, QuantileLoss):
                return len(loss.quantiles)
            elif isinstance(normalizer, NaNLabelEncoder):
                return len(normalizer.classes_)
            elif isinstance(loss, DistributionLoss):
                return len(loss.distribution_arguments)
            else:
                return 1  # default to 1

        # handle multiple targets
        new_kwargs = {}
        n_targets = len(dataset.target_names)
        if default_loss is None:
            default_loss = MAE()
        loss = kwargs.get("loss", default_loss)
        if n_targets > 1:  # try to infer number of ouput sizes
            if not isinstance(loss, MultiLoss):
                loss = MultiLoss([deepcopy(loss)] * n_targets)
                new_kwargs["loss"] = loss
            if isinstance(loss, MultiLoss) and "output_size" not in kwargs:
                new_kwargs["output_size"] = [
                    get_output_size(normalizer, l)
                    for normalizer, l in zip(dataset.target_normalizer.normalizers, loss.metrics)
                ]
        elif "output_size" not in kwargs:
            new_kwargs["output_size"] = get_output_size(dataset.target_normalizer, loss)
        return new_kwargs


    def size(self) -> int:
        """
        get number of parameters in model
        """
        return sum(p.numel() for p in self.parameters())


    def training_step(self, batch, batch_idx):
        """
        Train on batch.
        """
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        return log


    def training_epoch_end(self, outputs):
        self.epoch_end(outputs)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        return log


    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs)


    def test_step(self, batch, batch_idx):
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        return log


    def test_epoch_end(self, outputs):
        self.epoch_end(outputs)


    def create_log(
        self,
        x: Dict[str, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor],
        out: Dict[str, torch.Tensor],
        batch_idx: int,
        prediction_kwargs: Dict[str, Any] = {},
        quantiles_kwargs: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """
        Create the log used in the training and validation step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (Tuple[torch.Tensor, torch.Tensor]): y as passed to the loss function by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
            batch_idx (int): batch number
            prediction_kwargs (Dict[str, Any], optional): arguments to pass to
                :py:meth:`~pytorch_forcasting.models.base_model.BaseModel.to_prediction`. Defaults to {}.
            quantiles_kwargs (Dict[str, Any], optional):
                :py:meth:`~pytorch_forcasting.models.base_model.BaseModel.to_quantiles`. Defaults to {}.

        Returns:
            Dict[str, Any]: log dictionary to be returned by training and validation steps
        """
        # log
        if isinstance(self.loss, DistributionLoss):
            prediction_kwargs.setdefault("n_samples", 20)
            prediction_kwargs.setdefault("use_metric", True)
            quantiles_kwargs.setdefault("n_samples", 20)
            quantiles_kwargs.setdefault("use_metric", True)

        self.log_metrics(x, y, out, prediction_kwargs=prediction_kwargs)
        if self.log_interval > 0:
            self.log_prediction(
                x, out, batch_idx, prediction_kwargs=prediction_kwargs, quantiles_kwargs=quantiles_kwargs
            )
        return {}


    def step(
        self, x: Dict[str, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run for each train/val step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (Tuple[torch.Tensor, torch.Tensor]): y as passed to the loss function by the dataloader
            batch_idx (int): batch number
            **kwargs: additional arguments to pass to the network apart from ``x``

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: tuple where the first
                entry is a dictionary to which additional logging results can be added for consumption in the
                ``epoch_end`` hook and the second entry is the model's output.
        """
        # pack y sequence if different encoder lengths exist
        if (x["decoder_lengths"] < x["decoder_lengths"].max()).any():
            if isinstance(y[0], (list, tuple)):
                y = (
                    [
                        rnn.pack_padded_sequence(
                            y_part, lengths=x["decoder_lengths"].cpu(), batch_first=True, enforce_sorted=False
                        )
                        for y_part in y[0]
                    ],
                    y[1],
                )
            else:
                y = (
                    rnn.pack_padded_sequence(
                        y[0], lengths=x["decoder_lengths"].cpu(), batch_first=True, enforce_sorted=False
                    ),
                    y[1],
                )

        if self.training and len(self.hparams.monotone_constaints) > 0:
            # calculate gradient with respect to continous decoder features
            x["decoder_cont"].requires_grad_(True)
            assert not torch._C._get_cudnn_enabled(), (
                "To use monotone constraints, wrap model and training in context "
                "`torch.backends.cudnn.flags(enable=False)`"
            )
            out = self(x, **kwargs)
            prediction = out["prediction"]

            # handle multiple targets
            prediction_list = to_list(prediction)
            gradient = 0
            # todo: should monotone constrains be applicable to certain targets?
            for pred in prediction_list:
                gradient = (
                    gradient
                    + torch.autograd.grad(
                        outputs=pred,
                        inputs=x["decoder_cont"],
                        grad_outputs=torch.ones_like(pred),  # t
                        create_graph=True,  # allows usage in graph
                        allow_unused=True,
                    )[0]
                )

            # select relevant features
            indices = torch.tensor(
                [self.hparams.x_reals.index(name) for name in self.hparams.monotone_constaints.keys()]
            )
            monotonicity = torch.tensor(
                [val for val in self.hparams.monotone_constaints.values()], dtype=gradient.dtype, device=gradient.device
            )
            # add additionl loss if gradient points in wrong direction
            gradient = gradient[..., indices] * monotonicity[None, None]
            monotinicity_loss = gradient.clamp_max(0).mean()
            # multiply monotinicity loss by large number to ensure relevance and take to the power of 2
            # for smoothness of loss function
            monotinicity_loss = 10 * torch.pow(monotinicity_loss, 2)
            if isinstance(self.loss, (MASE, MultiLoss)):
                loss = self.loss(
                    prediction, y, encoder_target=x["encoder_target"], encoder_lengths=x["encoder_lengths"]
                )
            else:
                loss = self.loss(prediction, y)

            loss = loss * (1 + monotinicity_loss)
        else:
            out = self(x, **kwargs)

            # calculate loss
            prediction = out["prediction"]
            if isinstance(self.loss, (MASE, MultiLoss)):
                mase_kwargs = dict(encoder_target=x["encoder_target"], encoder_lengths=x["encoder_lengths"])
                loss = self.loss(prediction, y, **mase_kwargs)
            else:
                loss = self.loss(prediction, y)

        self.log(
            f"{self.current_stage}_loss",
            loss,
            on_step=self.training,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(x["decoder_target"]),
        )
        log = {"loss": loss, "n_samples": x["decoder_lengths"].size(0)}
        return log, out


    def log_metrics(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
        prediction_kwargs: Dict[str, Any] = None,
    ) -> None:
        """
        Log metrics every training/validation step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (torch.Tensor): y as passed to the loss function by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
            prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.
        """
        # logging losses - for each target
        if prediction_kwargs is None:
            prediction_kwargs = {}
        y_hat_point = self.to_prediction(out, **prediction_kwargs)
        if isinstance(self.loss, MultiLoss):
            y_hat_point_detached = [p.detach() for p in y_hat_point]
        else:
            y_hat_point_detached = [y_hat_point.detach()]

        for metric in self.logging_metrics:
            for idx, y_point, y_part, encoder_target in zip(
                list(range(len(y_hat_point_detached))),
                y_hat_point_detached,
                to_list(y[0]),
                to_list(x["encoder_target"]),
            ):
                y_true = (y_part, y[1])
                if isinstance(metric, MASE):
                    loss_value = metric(
                        y_point, y_true, encoder_target=encoder_target, encoder_lengths=x["encoder_lengths"]
                    )
                else:
                    loss_value = metric(y_point, y_true)
                if len(y_hat_point_detached) > 1:
                    target_tag = self.target_names[idx] + " "
                else:
                    target_tag = ""
                self.log(
                    f"{target_tag}{self.current_stage}_{metric.name}",
                    loss_value,
                    on_step=self.training,
                    on_epoch=True,
                    batch_size=len(x["decoder_target"]),
                )


    def forward(
        self, x: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Network forward pass.

        Args:
            x (Dict[str, Union[torch.Tensor, List[torch.Tensor]]]): network input (x as returned by the dataloader).
                See :py:meth:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet.to_dataloader` method that
                returns a tuple of ``x`` and ``y``. This function expects ``x``.

        Returns:
            NamedTuple[Union[torch.Tensor, List[torch.Tensor]]]: network outputs / dictionary of tensors or list
                of tensors. Create it using the
                :py:meth:`~pytorch_forecasting.models.base_model.BaseModel.to_network_output` method.
                The minimal required entries in the dictionary are (and shapes in brackets):

                * ``prediction`` (batch_size x n_decoder_time_steps x n_outputs or list thereof with each
                  entry for a different target): re-scaled predictions that can be fed to metric. List of tensors
                  if multiple targets are predicted at the same time.

                Before passing outputting the predictions, you want to rescale them into real space.
                By default, you can use the
                :py:meth:`~pytorch_forecasting.models.base_model.BaseModel.transform_output`
                method to achieve this.

        Example:

            .. code-block:: python

                def forward(self, x:
                    # x is a batch generated based on the TimeSeriesDataset, here we just use the
                    # continuous variables for the encoder
                    network_input = x["encoder_cont"].squeeze(-1)
                    prediction = self.linear(network_input)  #

                    # rescale predictions into target space
                    prediction = self.transform_output(prediction, target_scale=x["target_scale"])

                    # We need to return a dictionary that at least contains the prediction
                    # The parameter can be directly forwarded from the input.
                    # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
                    return self.to_network_output(prediction=prediction)



        """
        raise NotImplementedError()


    def epoch_end(self, outputs):
        """
        Run at epoch end for training or validation. Can be overriden in models.
        """
        pass


    @property
    def log_interval(self) -> float:
        """
        Log interval depending if training or validating
        """
        if self.training:
            return self.hparams.log_interval
        else:
            return self.hparams.log_val_interval

    def log_prediction(
        self, x: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], batch_idx: int, **kwargs
    ) -> None:
        """
        Log metrics every training/validation step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
            batch_idx (int): current batch index
            **kwargs: paramters to pass to ``plot_prediction``
        """
        # log single prediction figure
        if (batch_idx % self.log_interval == 0 or self.log_interval < 1.0) and self.log_interval > 0:
            if self.log_interval < 1.0:  # log multiple steps
                log_indices = torch.arange(
                    0, len(x["encoder_lengths"]), max(1, round(self.log_interval * len(x["encoder_lengths"])))
                )
            else:
                log_indices = [0]
            for idx in log_indices:
                fig = self.plot_prediction(x, out, idx=idx, add_loss_to_title=True, **kwargs)
                tag = f"{self.current_stage} prediction"
                if self.training:
                    tag += f" of item {idx} in global batch {self.global_step}"
                else:
                    tag += f" of item {idx} in batch {batch_idx}"
                if isinstance(fig, (list, tuple)):
                    for idx, f in enumerate(fig):
                        self.logger.experiment.add_figure(
                            f"{self.target_names[idx]} {tag}",
                            f,
                            global_step=self.global_step,
                        )
                else:
                    self.logger.experiment.add_figure(
                        tag,
                        fig,
                        global_step=self.global_step,
                    )


    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int = 0,
        add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
        show_future_observed: bool = True,
        ax=None,
        quantiles_kwargs: Dict[str, Any] = {},
        prediction_kwargs: Dict[str, Any] = {},
    ) -> plt.Figure:
        """
        Plot prediction of prediction vs actuals

        Args:
            x: network input
            out: network output
            idx: index of prediction to plot
            add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
                bool indicating if to use loss metric or tensor which contains losses for all samples.
                Calcualted losses are determined without weights. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on
            quantiles_kwargs (Dict[str, Any]): parameters for ``to_quantiles()`` of the loss metric.
            prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.

        Returns:
            matplotlib figure
        """
        # all true values for y of the first sample in batch
        encoder_targets = to_list(x["encoder_target"])
        decoder_targets = to_list(x["decoder_target"])

        y_raws = to_list(out["prediction"])  # raw predictions - used for calculating loss
        y_hats = to_list(self.to_prediction(out, **prediction_kwargs))
        y_quantiles = to_list(self.to_quantiles(out, **quantiles_kwargs))

        # for each target, plot
        figs = []
        for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
            y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
        ):

            y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
            max_encoder_length = x["encoder_lengths"].max()
            y = torch.cat(
                (
                    y_all[: x["encoder_lengths"][idx]],
                    y_all[max_encoder_length : (max_encoder_length + x["decoder_lengths"][idx])],
                ),
            )
            # move predictions to cpu
            y_hat = y_hat.detach().cpu()[idx, : x["decoder_lengths"][idx]]
            y_quantile = y_quantile.detach().cpu()[idx, : x["decoder_lengths"][idx]]
            y_raw = y_raw.detach().cpu()[idx, : x["decoder_lengths"][idx]]

            # move to cpu
            y = y.detach().cpu()
            # create figure
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
            n_pred = y_hat.shape[0]
            x_obs = np.arange(-(y.shape[0] - n_pred), 0)
            x_pred = np.arange(n_pred)
            prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
            obs_color = next(prop_cycle)["color"]
            pred_color = next(prop_cycle)["color"]
            # plot observed history
            if len(x_obs) > 0:
                if len(x_obs) > 1:
                    plotter = ax.plot
                else:
                    plotter = ax.scatter
                plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
            if len(x_pred) > 1:
                plotter = ax.plot
            else:
                plotter = ax.scatter

            # plot observed prediction
            if show_future_observed:
                plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

            # plot prediction
            plotter(x_pred, y_hat, label="predicted", c=pred_color)

            # plot predicted quantiles
            plotter(x_pred, y_quantile[:, y_quantile.shape[1] // 2], c=pred_color, alpha=0.15)
            for i in range(y_quantile.shape[1] // 2):
                if len(x_pred) > 1:
                    ax.fill_between(x_pred, y_quantile[:, i], y_quantile[:, -i - 1], alpha=0.15, fc=pred_color)
                else:
                    quantiles = torch.tensor([[y_quantile[0, i]], [y_quantile[0, -i - 1]]])
                    ax.errorbar(
                        x_pred,
                        y[[-n_pred]],
                        yerr=quantiles - y[-n_pred],
                        c=pred_color,
                        capsize=1.0,
                    )

            if add_loss_to_title is not False:
                if isinstance(add_loss_to_title, bool):
                    loss = self.loss
                elif isinstance(add_loss_to_title, torch.Tensor):
                    loss = add_loss_to_title.detach()[idx].item()
                elif isinstance(add_loss_to_title, Metric):
                    loss = add_loss_to_title
                else:
                    raise ValueError(f"add_loss_to_title '{add_loss_to_title}'' is unkown")
                if isinstance(loss, MASE):
                    loss_value = loss(y_raw[None], (y[-n_pred:][None], None), y[:n_pred][None])
                elif isinstance(loss, Metric):
                    try:
                        loss_value = loss(y_raw[None], (y[-n_pred:][None], None))
                    except Exception:
                        loss_value = "-"
                else:
                    loss_value = loss
                ax.set_title(f"Loss {loss_value}")
            ax.set_xlabel("Time index")
            fig.legend()
            figs.append(fig)

        # return multiple of target is a list, otherwise return single figure
        if isinstance(x["encoder_target"], (tuple, list)):
            return figs
        else:
            return fig


    def log_gradient_flow(self, named_parameters: Dict[str, torch.Tensor]) -> None:
        """
        log distribution of gradients to identify exploding / vanishing gradients
        """
        ave_grads = []
        layers = []
        for name, p in named_parameters:
            if p.grad is not None and p.requires_grad and "bias" not in name:
                layers.append(name)
                ave_grads.append(p.grad.abs().cpu().mean())
                self.logger.experiment.add_histogram(tag=name, values=p.grad, global_step=self.global_step)
        fig, ax = plt.subplots()
        ax.plot(ave_grads)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Average gradient")
        ax.set_yscale("log")
        ax.set_title("Gradient flow")
        self.logger.experiment.add_figure("Gradient flow", fig, global_step=self.global_step)


    def on_after_backward(self):
        """
        Log gradient flow for debugging.
        """
        if (
            self.hparams.log_interval > 0
            and self.global_step % self.hparams.log_interval == 0
            and self.hparams.log_gradient_flow
        ):
            self.log_gradient_flow(self.named_parameters())


    def configure_optimizers(self):
        """
        Configure optimizers.

        Uses single Ranger optimizer. Depending if learning rate is a list or a single float, implement dynamic
        learning rate scheduler or deterministic version

        Returns:
            Tuple[List]: first entry is list of optimizers and second is list of schedulers
        """
        # either set a schedule of lrs or find it dynamically
        if self.hparams.optimizer_params is None:
            optimizer_params = {}
        else:
            optimizer_params = self.hparams.optimizer_params
        # set optimizer
        lrs = self.hparams.learning_rate
        if isinstance(lrs, (list, tuple)):
            lr = lrs[0]
        else:
            lr = lrs
        if callable(self.optimizer):
            try:
                optimizer = self.optimizer(
                    self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            except TypeError:  # in case there is no weight decay
                optimizer = self.optimizer(self.parameters(), lr=lr, **optimizer_params)
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
            )
        elif self.hparams.optimizer == "ranger":
            optimizer = Ranger(self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
            )
        elif hasattr(torch.optim, self.hparams.optimizer):
            try:
                optimizer = getattr(torch.optim, self.hparams.optimizer)(
                    self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            except TypeError:  # in case there is no weight decay
                optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=lr, **optimizer_params)
        else:
            raise ValueError(f"Optimizer of self.hparams.optimizer={self.hparams.optimizer} unknown")

        # set scheduler
        if isinstance(lrs, (list, tuple)):  # change for each epoch
            # normalize lrs
            lrs = np.array(lrs) / lrs[0]
            scheduler_config = {
                "scheduler": LambdaLR(optimizer, lambda epoch: lrs[min(epoch, len(lrs) - 1)]),
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }
        elif self.hparams.reduce_on_plateau_patience is None:
            scheduler_config = {}
        else:  # find schedule based on validation loss
            scheduler_config = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=1.0 / self.hparams.reduce_on_plateau_reduction,
                    patience=self.hparams.reduce_on_plateau_patience,
                    cooldown=self.hparams.reduce_on_plateau_patience,
                    min_lr=self.hparams.reduce_on_plateau_min_lr,
                ),
                "monitor": "val_loss",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs) -> LightningModule:
        """
        Create model from dataset, i.e. save dataset parameters in model

        This function should be called as ``super().from_dataset()`` in a derived models that implement it

        Args:
            dataset (TimeSeriesDataSet): timeseries dataset

        Returns:
            BaseModel: Model that can be trained
        """
        if "output_transformer" not in kwargs:
            kwargs["output_transformer"] = dataset.target_normalizer
        net = cls(**kwargs)
        net.dataset_parameters = dataset.get_parameters()
        if dataset.multi_target:
            assert isinstance(
                net.loss, MultiLoss
            ), f"multiple targets require loss to be MultiLoss but found {net.loss}"
        else:
            assert not isinstance(net.loss, MultiLoss), "MultiLoss not compatible with single target"

        return net


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["dataset_parameters"] = getattr(
            self, "dataset_parameters", None
        )  # add dataset parameters for making fast predictions
        # hyper parameters are passed as arguments directly and not as single dictionary
        checkpoint["hparams_name"] = "kwargs"
        # save specials
        checkpoint[self.CHECKPOINT_HYPER_PARAMS_SPECIAL_KEY] = {k: getattr(self, k) for k in self.hparams_special}
        # add special hparams them back to save the hparams correctly for checkpoint
        checkpoint[self.CHECKPOINT_HYPER_PARAMS_KEY].update(checkpoint[self.CHECKPOINT_HYPER_PARAMS_SPECIAL_KEY])


    @property
    def target_names(self) -> List[str]:
        """
        List of targets that are predicted.

        Returns:
            List[str]: list of target names
        """
        if hasattr(self, "dataset_parameters") and self.dataset_parameters is not None:
            return to_list(self.dataset_parameters["target"])
        else:
            return [f"Target {idx + 1}" for idx in range(self.n_targets)]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.dataset_parameters = checkpoint.get("dataset_parameters", None)
        # load specials
        for k, v in checkpoint[self.CHECKPOINT_HYPER_PARAMS_SPECIAL_KEY].items():
            setattr(self, k, v)


    def to_prediction(self, out: Dict[str, Any], use_metric: bool = True, **kwargs):
        """
        Convert output to prediction using the loss metric.

        Args:
            out (Dict[str, Any]): output of network where "prediction" has been
                transformed with :py:meth:`~transform_output`
            use_metric (bool): if to use metric to convert for conversion, if False,
                simply take the average over ``out["prediction"]``
            **kwargs: arguments to metric ``to_quantiles`` method

        Returns:
            torch.Tensor: predictions of shape batch_size x timesteps
        """
        if not use_metric:
            # if samples were already drawn directly take mean
            # todo: support classification
            if isinstance(self.loss, MultiLoss):
                out = [Metric.to_prediction(loss, out["prediction"][idx]) for idx, loss in enumerate(self.loss)]
            else:
                out = Metric.to_prediction(self.loss, out["prediction"])
        else:
            try:
                out = self.loss.to_prediction(out["prediction"], **kwargs)
            except TypeError:  # in case passed kwargs do not exist
                out = self.loss.to_prediction(out["prediction"])
        return out


    def to_quantiles(self, out: Dict[str, Any], use_metric: bool = True, **kwargs):
        """
        Convert output to quantiles using the loss metric.

        Args:
            out (Dict[str, Any]): output of network where "prediction" has been
                transformed with :py:meth:`~transform_output`
            use_metric (bool): if to use metric to convert for conversion, if False,
                simply take the quantiles over ``out["prediction"]``
            **kwargs: arguments to metric ``to_quantiles`` method

        Returns:
            torch.Tensor: quantiles of shape batch_size x timesteps x n_quantiles
        """
        # if samples are output directly take quantiles
        if not use_metric:
            # todo: support classification
            if isinstance(self.loss, MultiLoss):
                out = [
                    Metric.to_quantiles(loss, out["prediction"][idx], quantiles=kwargs.get("quantiles", loss.quantiles))
                    for idx, loss in enumerate(self.loss)
                ]
            else:
                out = Metric.to_quantiles(
                    self.loss, out["prediction"], quantiles=kwargs.get("quantiles", self.loss.quantiles)
                )
        else:
            try:
                out = self.loss.to_quantiles(out["prediction"], **kwargs)
            except TypeError:  # in case passed kwargs do not exist
                out = self.loss.to_quantiles(out["prediction"])
        return out


    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        mode: Union[str, Tuple[str, str]] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        show_progress_bar: bool = False,
        return_x: bool = False,
        mode_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Run inference / prediction.

        Args:
            dataloader: dataloader, dataframe or dataset
            mode: one of "prediction", "quantiles", or "raw", or tuple ``("raw", output_name)`` where output_name is
                a name in the dictionary returned by ``forward()``
            return_index: if to return the prediction index (in the same order as the output, i.e. the row of the
                dataframe corresponds to the first dimension of the output and the given time index is the time index
                of the first prediction)
            return_decoder_lengths: if to return decoder_lengths (in the same order as the output
            batch_size: batch size for dataloader - only used if data is not a dataloader is passed
            num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
            fast_dev_run: if to only return results of first batch
            show_progress_bar: if to show progress bar. Defaults to False.
            return_x: if to return network inputs (in the same order as prediction output)
            mode_kwargs (Dict[str, Any]): keyword arguments for ``to_prediction()`` or ``to_quantiles()``
                for modes "prediction" and "quantiles"
            **kwargs: additional arguments to network's forward method

        Returns:
            output, x, index, decoder_lengths: some elements might not be present depending on what is configured
                to be returned
        """
        # convert to dataloader
        if isinstance(data, pd.DataFrame):
            data = TimeSeriesDataSet.from_parameters(self.dataset_parameters, data, predict=True)
        if isinstance(data, TimeSeriesDataSet):
            dataloader = data.to_dataloader(batch_size=batch_size, train=False, num_workers=num_workers)
        else:
            dataloader = data

        # mode kwargs default to None
        if mode_kwargs is None:
            mode_kwargs = {}

        # ensure passed dataloader is correct
        assert isinstance(dataloader.dataset, TimeSeriesDataSet), "dataset behind dataloader mut be TimeSeriesDataSet"

        # prepare model
        self.eval()  # no dropout, etc. no gradients

        # run predictions
        output = []
        decode_lenghts = []
        x_list = []
        index = []
        progress_bar = tqdm(desc="Predict", unit=" batches", total=len(dataloader), disable=not show_progress_bar)
        with torch.no_grad():
            for x, _ in dataloader:
                # move data to appropriate device
                data_device = x["encoder_cont"].device
                if data_device != self.device:
                    x = move_to_device(x, self.device)

                # make prediction
                out = self(x, **kwargs)  # raw output is dictionary

                lengths = x["decoder_lengths"]
                if return_decoder_lengths:
                    decode_lenghts.append(lengths)
                nan_mask = create_mask(lengths.max(), lengths)
                if isinstance(mode, (tuple, list)):
                    if mode[0] == "raw":
                        out = out[mode[1]]
                    else:
                        raise ValueError(
                            f"If a tuple is specified, the first element must be 'raw' - got {mode[0]} instead"
                        )
                elif mode == "prediction":
                    out = self.to_prediction(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask, torch.tensor(float("nan"))) if o.dtype == torch.float else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:  # only floats can be filled with nans
                        out = out.masked_fill(nan_mask, torch.tensor(float("nan")))
                elif mode == "quantiles":
                    out = self.to_quantiles(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                            if o.dtype == torch.float
                            else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:
                        out = out.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                elif mode == "raw":
                    pass
                else:
                    raise ValueError(f"Unknown mode {mode} - see docs for valid arguments")

                out = move_to_device(out, device="cpu")

                output.append(out)
                if return_x:
                    x = move_to_device(x, "cpu")
                    x_list.append(x)
                if return_index:
                    index.append(dataloader.dataset.x_to_index(x))
                progress_bar.update()
                if fast_dev_run:
                    break

        # concatenate output (of different batches)
        if isinstance(mode, (tuple, list)) or mode != "raw":
            if isinstance(output[0], (tuple, list)) and len(output[0]) > 0 and isinstance(output[0][0], torch.Tensor):
                output = [_torch_cat_na([out[idx] for out in output]) for idx in range(len(output[0]))]
            else:
                output = _torch_cat_na(output)
        elif mode == "raw":
            output = _concatenate_output(output)

        # generate output
        if return_x or return_index or return_decoder_lengths:
            output = [output]
        if return_x:
            output.append(_concatenate_output(x_list))
        if return_index:
            output.append(pd.concat(index, axis=0, ignore_index=True))
        if return_decoder_lengths:
            output.append(torch.cat(decode_lenghts, dim=0))
        return output


    def predict_dependency(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        variable: str,
        values: Iterable,
        mode: str = "dataframe",
        target="decoder",
        show_progress_bar: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor, pd.Series, pd.DataFrame]:
        """
        Predict partial dependency.

        Args:
            data (Union[DataLoader, pd.DataFrame, TimeSeriesDataSet]): data
            variable (str): variable which to modify
            values (Iterable): array of values to probe
            mode (str, optional): Output mode. Defaults to "dataframe". Either

                * "series": values are average prediction and index are probed values
                * "dataframe": columns are as obtained by the `dataset.x_to_index()` method,
                    prediction (which is the mean prediction over the time horizon),
                    normalized_prediction (which are predictions devided by the prediction for the first probed value)
                    the variable name for the probed values
                * "raw": outputs a tensor of shape len(values) x prediction_shape

            target: Defines which values are overwritten for making a prediction.
                Same as in :py:meth:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet.set_overwrite_values`.
                Defaults to "decoder".
            show_progress_bar: if to show progress bar. Defaults to False.
            **kwargs: additional kwargs to :py:meth:`~predict` method

        Returns:
            Union[np.ndarray, torch.Tensor, pd.Series, pd.DataFrame]: output
        """
        values = np.asarray(values)
        if isinstance(data, pd.DataFrame):  # convert to dataframe
            data = TimeSeriesDataSet.from_parameters(self.dataset_parameters, data, predict=True)
        elif isinstance(data, DataLoader):
            data = data.dataset

        results = []
        progress_bar = tqdm(desc="Predict", unit=" batches", total=len(values), disable=not show_progress_bar)
        for idx, value in enumerate(values):
            # set values
            data.set_overwrite_values(variable=variable, values=value, target=target)
            # predict
            kwargs.setdefault("mode", "prediction")

            if idx == 0 and mode == "dataframe":  # need index for returning as dataframe
                res, index = self.predict(data, return_index=True, **kwargs)
                results.append(res)
            else:
                results.append(self.predict(data, **kwargs))
            # increment progress
            progress_bar.update()

        data.reset_overwrite_values()  # reset overwrite values to avoid side-effect

        # results to one tensor
        results = torch.stack(results, dim=0)

        # convert results to requested output format
        if mode == "series":
            results = results[:, ~torch.isnan(results[0])].mean(1)  # average samples and prediction horizon
            results = pd.Series(results, index=values)

        elif mode == "dataframe":
            # take mean over time
            is_nan = torch.isnan(results)
            results[is_nan] = 0
            results = results.sum(-1) / (~is_nan).float().sum(-1)

            # create dataframe
            dependencies = (
                index.iloc[np.tile(np.arange(len(index)), len(values))]
                .reset_index(drop=True)
                .assign(prediction=results.flatten())
            )
            dependencies[variable] = values.repeat(len(data))
            first_prediction = dependencies.groupby(data.group_ids, observed=True).prediction.transform("first")
            dependencies["normalized_prediction"] = dependencies["prediction"] / first_prediction
            dependencies["id"] = dependencies.groupby(data.group_ids, observed=True).ngroup()
            results = dependencies

        elif mode == "raw":
            pass

        else:
            raise ValueError(f"mode {mode} is unknown - see documentation for available modes")

        return results