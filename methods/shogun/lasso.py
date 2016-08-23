'''
  @file lasso.py
  @author Youssef Emad El-Din

  Lasso Regression with shogun.
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
from modshogun import RegressionLabels, RealFeatures
from modshogun import LeastAngleRegression

'''
This class implements the Lasso Regression benchmark.
'''
class LASSO(object):

  '''
  Create the Lasso Regression benchmark instance.

  @param dataset - Input dataset to perform Lasso Regression on.
  @param timeout - The time until the timeout. Default no timeout.
  @param verbose - Display informational messages.
  '''
  def __init__(self, dataset, timeout=0, verbose=True):
    self.verbose = verbose
    self.dataset = dataset
    self.timeout = timeout

  '''
  Use the shogun libary to implement Linear Regression.

  @param options - Extra options for the method.
  @return - Elapsed time in seconds or a negative value if the method was not
  successful.
  '''
  def LASSOShogun(self, options):
    def RunLASSOShogun(q):
      totalTimer = Timer()

      # Load input dataset.
      # If the dataset contains two files then the second file is the responses
      # file.
      try:
        # Load input dataset.
        Log.Info("Loading dataset", self.verbose)
        inputData = np.genfromtxt(self.dataset[0], delimiter=',')
        responsesData = np.genfromtxt(self.dataset[1], delimiter=',')

        # Get all the parameters.
        lambda1 = re.search("-l (\d+\.\d+)", options)
        lambda1 = 0.0 if not lambda1 else float(lambda1.group(1))

        with totalTimer:
          model = LeastAngleRegression(True)
          model.set_labels(RegressionLabels(responsesData))
          model.train(RealFeatures(inputData.T))

      except Exception as e:
        print(e)
        q.put(-1)
        return -1

      time = totalTimer.ElapsedTime()
      q.put(time)
      return time

    return timeout(RunLASSOShogun, self.timeout)

  '''
  Perform Lasso Regression. If the method has been successfully completed
  return the elapsed time in seconds.

  @param options - Extra options for the method.
  @return - Elapsed time in seconds or a negative value if the method was not
  successful.
  '''
  def RunMetrics(self, options):
    Log.Info("Perform Lasso Regression.", self.verbose)
    
    if len(self.dataset) != 2:
      Log.Fatal("This method requires two datasets.")
      return -1

    results = self.LASSOShogun(options)
    if results < 0:
      return results

    metrics = {'Runtime' : results}

    return metrics
