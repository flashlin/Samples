import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import scipy.stats
import torch
from torch import nn
from pytorch_lightning import LightningModule
from pytorch_forecasting.data.encoders import EncoderNormalizer, GroupNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.metrics.base_metrics import Metric
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

from BaseModel import BaseModel

class AutoRegressiveBaseModel(BaseModel):
    """
    Model with additional methods for autoregressive models.

    Adds in particular the :py:meth:`~decode_autoregressive` method for making auto-regressive predictions.

    Assumes the following hyperparameters:

    Args:
        target (str): name of target variable
        target_lags (Dict[str, Dict[str, int]]): dictionary of target names mapped each to a dictionary of corresponding
            lagged variables and their lags.
            Lags can be useful to indicate seasonality to the models. If you know the seasonalit(ies) of your data,
            add at least the target variables with the corresponding lags to improve performance.
            Defaults to no lags, i.e. an empty dictionary.
    """

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        **kwargs,
    ) -> LightningModule:
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            LightningModule
        """
        kwargs.setdefault("target", dataset.target)
        # check that lags for targets are the same
        lags = {name: lag for name, lag in dataset.lags.items() if name in dataset.target_names}  # filter for targets
        target0 = dataset.target_names[0]
        lag = set(lags.get(target0, []))
        for target in dataset.target_names:
            assert lag == set(lags.get(target, [])), f"all target lags in dataset must be the same but found {lags}"

        kwargs.setdefault("target_lags", {name: dataset._get_lagged_names(name) for name in lags})
        return super().from_dataset(dataset, **kwargs)


    def output_to_prediction(
        self,
        normalized_prediction_parameters: torch.Tensor,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_samples: int = 1,
        **kwargs,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Convert network output to rescaled and normalized prediction.

        Function is typically not called directly but via :py:meth:`~decode_autoregressive`.

        Args:
            normalized_prediction_parameters (torch.Tensor): network prediction output
            target_scale (Union[List[torch.Tensor], torch.Tensor]): target scale to rescale network output
            n_samples (int, optional): Number of samples to draw independently. Defaults to 1.
            **kwargs: extra arguments for dictionary passed to :py:meth:`~transform_output` method.

        Returns:
            Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]: tuple of rescaled prediction and
                normalized prediction (e.g. for input into next auto-regressive step)
        """
        single_prediction = to_list(normalized_prediction_parameters)[0].ndim == 2
        if single_prediction:  # add time dimension as it is expected
            normalized_prediction_parameters = apply_to_list(normalized_prediction_parameters, lambda x: x.unsqueeze(1))
        # transform into real space
        prediction_parameters = self.transform_output(
            prediction=normalized_prediction_parameters, target_scale=target_scale, **kwargs
        )
        # todo: handle classification
        # sample value(s) from distribution and  select first sample
        if isinstance(self.loss, DistributionLoss) or (
            isinstance(self.loss, MultiLoss) and isinstance(self.loss[0], DistributionLoss)
        ):
            # todo: handle mixed losses
            if n_samples > 1:
                prediction_parameters = apply_to_list(
                    prediction_parameters, lambda x: x.reshape(int(x.size(0) / n_samples), n_samples, -1)
                )
                prediction = self.loss.sample(prediction_parameters, 1)
                prediction = apply_to_list(prediction, lambda x: x.reshape(x.size(0) * n_samples, 1, -1))
            else:
                prediction = self.loss.sample(normalized_prediction_parameters, 1)

        else:
            prediction = prediction_parameters
        # normalize prediction prediction
        normalized_prediction = self.output_transformer.transform(prediction, target_scale=target_scale)
        if isinstance(normalized_prediction, list):
            input_target = torch.cat(normalized_prediction, dim=-1)
        else:
            input_target = normalized_prediction  # set next input target to normalized prediction

        # remove time dimension
        if single_prediction:
            prediction = apply_to_list(prediction, lambda x: x.squeeze(1))
            input_target = input_target.squeeze(1)
        return prediction, input_target


    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        first_hidden_state: Any,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        n_samples: int = 1,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Make predictions in auto-regressive manner.

        Supports only continuous targets.

        Args:
            decode_one (Callable): function that takes at least the following arguments:

                * ``idx`` (int): index of decoding step (from 0 to n_decoder_steps-1)
                * ``lagged_targets`` (List[torch.Tensor]): list of normalized targets.
                  List is ``idx + 1`` elements long with the most recent entry at the end, i.e.
                  ``previous_target = lagged_targets[-1]`` and in general ``lagged_targets[-lag]``.
                * ``hidden_state`` (Any): Current hidden state required for prediction.
                  Keys are variable names. Only lags that are greater than ``idx`` are included.
                * additional arguments are not dynamic but can be passed via the ``**kwargs`` argument

                And returns tuple of (not rescaled) network prediction output and hidden state for next
                auto-regressive step.

            first_target (Union[List[torch.Tensor], torch.Tensor]): first target value to use for decoding
            first_hidden_state (Any): first hidden state used for decoding
            target_scale (Union[List[torch.Tensor], torch.Tensor]): target scale as in ``x``
            n_decoder_steps (int): number of decoding/prediction steps
            n_samples (int): number of independent samples to draw from the distribution -
                only relevant for multivariate models. Defaults to 1.
            **kwargs: additional arguments that are passed to the decode_one function.

        Returns:
            Union[List[torch.Tensor], torch.Tensor]: re-scaled prediction

        Example:

            LSTM/GRU decoder

            .. code-block:: python

                def decode(self, x, hidden_state):
                    # create input vector
                    input_vector = x["decoder_cont"].clone()
                    input_vector[..., self.target_positions] = torch.roll(
                        input_vector[..., self.target_positions],
                        shifts=1,
                        dims=1,
                    )
                    # but this time fill in missing target from encoder_cont at the first time step instead of
                    # throwing it away
                    last_encoder_target = x["encoder_cont"][
                        torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
                        x["encoder_lengths"] - 1,
                        self.target_positions.unsqueeze(-1)
                    ].T.contiguous()
                    input_vector[:, 0, self.target_positions] = last_encoder_target

                    if self.training:  # training mode
                        decoder_output, _ = self.rnn(
                            x,
                            hidden_state,
                            lengths=x["decoder_lengths"],
                            enforce_sorted=False,
                        )

                        # from hidden state size to outputs
                        if isinstance(self.hparams.target, str):  # single target
                            output = self.distribution_projector(decoder_output)
                        else:
                            output = [projector(decoder_output) for projector in self.distribution_projector]

                        # predictions are not yet rescaled -> so rescale now
                        return self.transform_output(output, target_scale=target_scale)

                    else:  # prediction mode
                        target_pos = self.target_positions

                        def decode_one(idx, lagged_targets, hidden_state):
                            x = input_vector[:, [idx]]
                            x[:, 0, target_pos] = lagged_targets[-1]  # overwrite at target positions

                            # overwrite at lagged targets positions
                            for lag, lag_positions in lagged_target_positions.items():
                                if idx > lag:  # only overwrite if target has been generated
                                    x[:, 0, lag_positions] = lagged_targets[-lag]

                            decoder_output, hidden_state = self.rnn(x, hidden_state)
                            decoder_output = decoder_output[:, 0]  # take first timestep
                            # from hidden state size to outputs
                            if isinstance(self.hparams.target, str):  # single target
                                output = self.distribution_projector(decoder_output)
                            else:
                                output = [projector(decoder_output) for projector in self.distribution_projector]
                            return output, hidden_state

                        # make predictions which are fed into next step
                        output = self.decode_autoregressive(
                            decode_one,
                            first_target=input_vector[:, 0, target_pos],
                            first_hidden_state=hidden_state,
                            target_scale=x["target_scale"],
                            n_decoder_steps=input_vector.size(1),
                        )

                        # predictions are already rescaled
                        return output

        """
        # make predictions which are fed into next step
        output = []
        current_target = first_target
        current_hidden_state = first_hidden_state

        normalized_output = [first_target]

        for idx in range(n_decoder_steps):
            # get lagged targets
            current_target, current_hidden_state = decode_one(
                idx, lagged_targets=normalized_output, hidden_state=current_hidden_state, **kwargs
            )

            # get prediction and its normalized version for the next step
            prediction, current_target = self.output_to_prediction(
                current_target, target_scale=target_scale, n_samples=n_samples
            )
            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples

            output.append(prediction)
        if isinstance(self.hparams.target, str):
            output = torch.stack(output, dim=1)
        else:
            # for multi-targets
            output = [torch.stack([out[idx] for out in output], dim=1) for idx in range(len(self.target_positions))]
        return output


    @property
    def target_positions(self) -> torch.LongTensor:
        """
        Positions of target variable(s) in covariates.

        Returns:
            torch.LongTensor: tensor of positions.
        """
        # todo: expand for categorical targets
        return torch.tensor(
            [0],
            device=self.device,
            dtype=torch.long,
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

        # get predictions
        if isinstance(self.loss, DistributionLoss):
            prediction_kwargs.setdefault("use_metric", False)
            quantiles_kwargs.setdefault("use_metric", False)

        return super().plot_prediction(
            x=x,
            out=out,
            idx=idx,
            add_loss_to_title=add_loss_to_title,
            show_future_observed=show_future_observed,
            ax=ax,
            quantiles_kwargs=quantiles_kwargs,
            prediction_kwargs=prediction_kwargs,
        )


    @property
    def lagged_target_positions(self) -> Dict[int, torch.LongTensor]:
        """
        Positions of lagged target variable(s) in covariates.

        Returns:
            Dict[int, torch.LongTensor]: dictionary mapping integer lags to tensor of variable positions.
        """
        raise Exception(
            "lagged targets can only be used with class inheriting "
            "from AutoRegressiveBaseModelWithCovariates but not from AutoRegressiveBaseModel"
        )