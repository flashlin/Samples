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
from BaseModelWithCovariates import BaseModelWithCovariates
from AutoRegressiveBaseModel import AutoRegressiveBaseModel

class AutoRegressiveBaseModelWithCovariates(BaseModelWithCovariates, AutoRegressiveBaseModel):
    """
    Model with additional methods for autoregressive models with covariates.

    Assumes the following hyperparameters:

    Args:
        target (str): name of target variable
        target_lags (Dict[str, Dict[str, int]]): dictionary of target names mapped each to a dictionary of corresponding
            lagged variables and their lags.
            Lags can be useful to indicate seasonality to the models. If you know the seasonalit(ies) of your data,
            add at least the target variables with the corresponding lags to improve performance.
            Defaults to no lags, i.e. an empty dictionary.
        static_categoricals (List[str]): names of static categorical variables
        static_reals (List[str]): names of static continuous variables
        time_varying_categoricals_encoder (List[str]): names of categorical variables for encoder
        time_varying_categoricals_decoder (List[str]): names of categorical variables for decoder
        time_varying_reals_encoder (List[str]): names of continuous variables for encoder
        time_varying_reals_decoder (List[str]): names of continuous variables for decoder
        x_reals (List[str]): order of continuous variables in tensor passed to forward function
        x_categoricals (List[str]): order of categorical variables in tensor passed to forward function
        embedding_sizes (Dict[str, Tuple[int, int]]): dictionary mapping categorical variables to tuple of integers
            where the first integer denotes the number of categorical classes and the second the embedding size
        embedding_labels (Dict[str, List[str]]): dictionary mapping (string) indices to list of categorical labels
        embedding_paddings (List[str]): names of categorical variables for which label 0 is always mapped to an
             embedding vector filled with zeros
        categorical_groups (Dict[str, List[str]]): dictionary of categorical variables that are grouped together and
            can also take multiple values simultaneously (e.g. holiday during octoberfest). They should be implemented
            as bag of embeddings
    """

    @property
    def lagged_target_positions(self) -> Dict[int, torch.LongTensor]:
        """
        Positions of lagged target variable(s) in covariates.

        Returns:
            Dict[int, torch.LongTensor]: dictionary mapping integer lags to tensor of variable positions.
        """
        # todo: expand for categorical targets
        if len(self.hparams.target_lags) == 0:
            return {}
        else:
            # extract lags which are the same across all targets
            lags = list(next(iter(self.hparams.target_lags.values())).values())
            lag_names = {l: [] for l in lags}
            for targeti_lags in self.hparams.target_lags.values():
                for name, l in targeti_lags.items():
                    lag_names[l].append(name)

            lag_pos = {
                lag: torch.tensor(
                    [self.hparams.x_reals.index(name) for name in to_list(names)],
                    device=self.device,
                    dtype=torch.long,
                )
                for lag, names in lag_names.items()
            }
            return lag_pos