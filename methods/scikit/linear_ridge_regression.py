'''
  @file linear_ridge_regression.py
  @author Ali Mohamed

  Linear Ridge Regression with scikit.
'''

import os
import sys
import inspect

# Import the util path, this method even works if the path contains symlinks to
# modules.
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(
  os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../util")))
if cmd_subfolder not in sys.path:
  sys.path.insert(0, cmd_subfolder)

#Import the metrics definitions path.
metrics_folder = os.path.realpath(os.path.abspath(os.path.join(
  os.path.split(inspect.getfile(inspect.currentframe()))[0], "../metrics")))
if metrics_folder not in sys.path:
  sys.path.insert(0, metrics_folder)

from log import *
from timer import *
from definitions import *
from misc import *

import numpy as np
from sklearn.linear_model import Ridge

'''
This class implements the Linear Ridge Regression benchmark.
'''
class LinearRidgeRegression(object):

  '''
  Create the Linear Ridge Regression benchmark instance.

  @param dataset - Input dataset to perform Linear Ridge Regression on.
  @param timeout - The time until the timeout. Default no timeout.
  @param verbose - Display informational messages.
  '''
  def __init__(self, dataset, timeout=0, verbose=True):
    self.verbose = verbose
    self.dataset = dataset
    self.timeout = timeout

  '''
  Build the model for the Linear Ridge Regression.

  @param data - The train data.
  @param responses - The responses for the training set.
  @return The created model.
  '''
  def BuildModel(self, data, responses, alpha=1.0):
    # Create and train the classifier.
    lrr = Ridge(alpha=alpha)
    lrr.fit(data, responses)
    return lrr

  '''
  Use the scikit libary to implement Linear Ridge Regression.

  @param options - Extra options for the method.
  @return - Elapsed time in seconds or a negative value if the method was not
  successful.
  '''
  def LinearRidgeRegressionScikit(self, options):
    def RunLinearRidgeRegressionScikit(q):
      totalTimer = Timer()

      # Load input dataset.
      # If the dataset contains two files then the second file is the test file.
      Log.Info("Loading dataset", self.verbose)
      if len(self.dataset) >= 2:
        testSet = LoadDataset(self.dataset[1])

      # Use the last row of the training set as the responses.
      X, y = SplitTrainData(self.dataset)
      alpha = re.search("-t (\d+)", options)
      alpha = 1.0 if not alpha else int(alpha.group(1))

      try:
        with totalTimer:
          # Perform linear ridge regression.
          model = self.BuildModel(X,y, alpha=alpha)

          if len(self.dataset) >= 2:
            model.predict(testSet)

      except Exception as e:
        q.put(-1)
        return -1

      time = totalTimer.ElapsedTime()
      q.put(time)
      return time

    return timeout(RunLinearRidgeRegressionScikit, self.timeout)

  '''
  Perform Linear Ridge Regression. If the method has been successfully completed
  return the elapsed time in seconds.

  @param options - Extra options for the method.
  @return - Elapsed time in seconds or a negative value if the method was not
  successful.
  '''
  def RunTiming(self, options):
    Log.Info("Perform Linear Ridge Regression.", self.verbose)
    return self.LinearRidgeRegressionScikit(options)

  '''
  Run all the metrics for Linear Ridge Regression.
  '''
  def RunMetrics(self, options):
    if len(self.dataset) >= 3:

      trainData, labels = SplitTrainData(self.dataset)

      testData = LoadDataset(self.dataset[1])
      truelabels = LoadDataset(self.dataset[2])
      alpha = re.search("-t (\d+)", options)
      alpha = 1.0 if not alpha else int(alpha.group(1))

      predictedlabels = np.rint(self.BuildModel(trainData, labels, alpha=alpha).predict(testData))

      SimpleMSE = Metrics.SimpleMeanSquaredError(truelabels, predictedlabels)
      metrics_dict = {}
      metrics_dict['Simple MSE'] = SimpleMSE
      return metrics_dict

    else:
      Log.Fatal("This method requires three datasets.")
