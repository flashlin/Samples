import torch
from torch import nn
import numpy as np
import pandas as pd
from FullyConnectedClassificationModel import FullyConnectedClassificationModel
from FullyConnectedForDistributionLossModel import FullyConnectedForDistributionLossModel
from FullyConnectedModelWithCovariates import FullyConnectedModelWithCovariates
from FullyConnectedModule import FullyConnectedModel, FullyConnectedModule
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import WeightedRandomSampler
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, MultiLoss
from FullyConnectedMultiOutputModule import FullyConnectedMultiOutputModule
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


from FullyConnectedMultiTargetModel import FullyConnectedMultiTargetModel
from LSTMModel import LSTMModel


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def info(message):
   print(f"{bcolors.OKGREEN}{message}{bcolors.ENDC}")


def test_that__network_works_as_intended():
   network = FullyConnectedModule(input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2)
   x = torch.rand(20, 5)
   info(f"{network(x).shape=}")

def generate_test_data():
   test_data = pd.DataFrame(
      dict(
        value=np.random.rand(30) - 0.5,
        group=np.repeat(np.arange(3), 10),
        time_idx=np.tile(np.arange(10), 3),
      )
   )
   info(f"test_data=")
   print(f"{test_data}")
   return test_data


def create_the_dataset(test_data: pd.DataFrame):
   # create the dataset from the pandas dataframe
   dataset = TimeSeriesDataSet(
       test_data,
       group_ids=["group"],
       target="value",
       time_idx="time_idx",
       min_encoder_length=5,
       max_encoder_length=5,
       min_prediction_length=2,
       max_prediction_length=2,
       time_varying_unknown_reals=["value"],
   )
   info("dataset.get_parameters")
   print(f"{dataset.get_parameters()}")
   return dataset



def test():
   test_that__network_works_as_intended()
   test_data = generate_test_data()
   dataset = create_the_dataset(test_data)
   # convert the dataset to a dataloader
   dataloader = dataset.to_dataloader(batch_size=4)
   # and load the first batch
   x, y = next(iter(dataloader))
   print("x =", x)
   print("\ny =", y)
   print("\nsizes of x =")
   for key, value in x.items():
       print(f"\t{key} = {value.size()}")
   model = FullyConnectedModel.from_dataset(dataset, input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2)
   x, y = next(iter(dataloader))
   output = model(x)
   info("output")
   print(output)
   print(dataset.x_to_index(x))
   info("Coupling datasets and models")
   model = FullyConnectedModel.from_dataset(dataset, hidden_size=10, n_hidden_layers=2)
   from torchsummary import summary
   #summary(model)
   #model.summarize("full")  # print model summary
   info("hparams")
   print(model.hparams)
   classification_test_data = pd.DataFrame(
      dict(
        target=np.random.choice(["A", "B", "C"], size=30),  # CHANGING values to predict to a categorical
        value=np.random.rand(30),  # INPUT values - see next section on covariates how to use categorical inputs
        group=np.repeat(np.arange(3), 10),
        time_idx=np.tile(np.arange(10), 3),
      )
   )
   info("classification_test_data")
   print(classification_test_data)
   # create the dataset from the pandas dataframe
   classification_dataset = TimeSeriesDataSet(
      classification_test_data,
      group_ids=["group"],
      target="target",  # SWITCHING to categorical target
      time_idx="time_idx",
      min_encoder_length=5,
      max_encoder_length=5,
      min_prediction_length=2,
      max_prediction_length=2,
      time_varying_unknown_reals=["value"],
      target_normalizer=NaNLabelEncoder(),  # Use the NaNLabelEncoder to encode categorical target
   )
   x, y = next(iter(classification_dataset.to_dataloader(batch_size=4)))
   print(y[0])  # target values are encoded categories
   ###
   model = FullyConnectedClassificationModel.from_dataset(classification_dataset, hidden_size=10, n_hidden_layers=2)
   # model.summarize("full")
   info("full hparams")
   print(model.hparams)
   # passing x through model
   print(model(x)["prediction"].shape)
   # Predicting multiple targets at the same time
   multi_target_test_data = pd.DataFrame(
    dict(
        target1=np.random.rand(30),
        target2=np.random.rand(30),
        group=np.repeat(np.arange(3), 10),
        time_idx=np.tile(np.arange(10), 3),
    )
   )
   print(multi_target_test_data)
   # create the dataset from the pandas dataframe
   multi_target_dataset = TimeSeriesDataSet(
       multi_target_test_data,
       group_ids=["group"],
       target=["target1", "target2"],  # USING two targets
       time_idx="time_idx",
       min_encoder_length=5,
       max_encoder_length=5,
       min_prediction_length=2,
       max_prediction_length=2,
       time_varying_unknown_reals=["target1", "target2"],
       target_normalizer=MultiNormalizer(
           [EncoderNormalizer(), TorchNormalizer()]
       ),  # Use the NaNLabelEncoder to encode categorical target
   )

   x, y = next(iter(multi_target_dataset.to_dataloader(batch_size=4)))
   print(y[0])  # target values are a list of targets
   ######
   from pytorch_forecasting.metrics import MAE, SMAPE, MultiLoss
   from pytorch_forecasting.utils import to_list
   model = FullyConnectedMultiTargetModel.from_dataset(
    multi_target_dataset,
    hidden_size=10,
    n_hidden_layers=2,
    loss=MultiLoss(metrics=[MAE(), SMAPE()], weights=[2.0, 1.0]),
   )
   # model.summarize("full")
   print(model.hparams)
   #####
   from pytorch_forecasting.models.base_model import BaseModelWithCovariates
   print(BaseModelWithCovariates.__doc__)
   test_data_with_covariates = pd.DataFrame(
      dict(
         # as before
         value=np.random.rand(30),
         group=np.repeat(np.arange(3), 10),
         time_idx=np.tile(np.arange(10), 3),
         # now adding covariates
         categorical_covariate=np.random.choice(["a", "b"], size=30),
         real_covariate=np.random.rand(30),
      )
   ).astype(
      dict(group=str)
   )  # categorical covariates have to be of string type
   info("BaseModelWithCovariates")
   print(test_data_with_covariates)
   # create the dataset from the pandas dataframe
   dataset_with_covariates = TimeSeriesDataSet(
       test_data_with_covariates,
       group_ids=["group"],
       target="value",
       time_idx="time_idx",
       min_encoder_length=5,
       max_encoder_length=5,
       min_prediction_length=2,
       max_prediction_length=2,
       time_varying_unknown_reals=["value"],
       time_varying_known_reals=["real_covariate"],
       time_varying_known_categoricals=["categorical_covariate"],
       static_categoricals=["group"],
   )

   model = FullyConnectedModelWithCovariates.from_dataset(dataset_with_covariates, hidden_size=10, n_hidden_layers=2)
   # model.summarize("full")  # print model summary
   print(model.hparams)
   x, y = next(iter(dataset_with_covariates.to_dataloader(batch_size=4)))  # generate batch
   output = model(x)  # pass batch through model
   print(output)
   #### Implementing an autoregressive / recurrent model
   model = LSTMModel.from_dataset(dataset, n_layers=2, hidden_size=10)
   #model.summarize("full")
   print(model.hparams)
   x, y = next(iter(dataloader))
   print(
       "prediction shape in training:", model(x)["prediction"].size()
   )  # batch_size x decoder time steps x 1 (1 for one target dimension)
   model.eval()  # set model into eval mode to use autoregressive prediction
   print("prediction shape in inference:", model(x)["prediction"].size())  # should be the same as in training
   ### Using and defining a custom/non-trivial metric
   from pytorch_forecasting.metrics import MAE
   model = FullyConnectedModel.from_dataset(dataset, hidden_size=10, n_hidden_layers=2, loss=MAE())
   print(model.hparams)
   ###
   # test that network works as intended
   network = FullyConnectedMultiOutputModule(input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2, n_outputs=7)
   print(network(torch.rand(20, 5)).shape)  # <<<<<<<<<< instead of shape (20, 2), returning additional dimension for quantiles
   ####
   model = FullyConnectedForDistributionLossModel.from_dataset(dataset, hidden_size=10, n_hidden_layers=2)
   #model.summarize("full")
   print(model.hparams)
   print(x["decoder_lengths"])
   x, y = next(iter(dataloader))
   print("parameter predition shape: ", model(x)["prediction"].size())
   model.eval()  # set model into eval mode for sampling
   print("sample prediction shape: ", model(x, n_samples=200)["prediction"].size())

test()


