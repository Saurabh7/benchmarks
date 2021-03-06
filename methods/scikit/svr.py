'''
  @file SVR.py
  @author Saurabh Mahindre

  SVR Regression with scikit.
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

from log import *
from timer import *
from misc import *

import numpy as np
from sklearn.svm import SVR as SSVR

'''
This class implements the SVR Regression benchmark.
'''
class SVR(object):

  '''
  Create the SVR Regression benchmark instance.

  @param dataset - Input dataset to perform Least Angle Regression on.
  @param timeout - The time until the timeout. Default no timeout.
  @param verbose - Display informational messages.
  '''
  def __init__(self, dataset, timeout=0, verbose=True):
    self.verbose = verbose
    self.dataset = dataset
    self.timeout = timeout
   

  '''
  Use the scikit libary to implement svr Regression.

  @param options - Extra options for the method.
  @return - Elapsed time in seconds or a negative value if the method was not
  successful.
  '''
  def SVRScikit(self, options):
    def RunSVRScikit(q):
      totalTimer = Timer()

      # Load input dataset.
      Log.Info("Loading dataset", self.verbose)
      # Use the last row of the training set as the responses.
      X, y = SplitTrainData(self.dataset)

      # Get all the parameters.
      c = re.search("-c (\d+\.\d+)", options)
      e = re.search("-e (\d+\.\d+)", options)
      g = re.search("-g (\d+\.\d+)", options)

      C = 1.0 if not c else float(c.group(1))
      epsilon = 1.0 if not e else float(e.group(1))
      gamma = 0.1 if not g else float(g.group(1))

      try:
        with totalTimer:
          # Perform SVR.
          model = SSVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
          model.fit(X, y)
      except Exception as e:
        q.put(-1)
        return -1
      
      time = totalTimer.ElapsedTime()
      q.put(time)
      return time

    return timeout(RunSVRScikit, self.timeout)

  '''
  Perform SVR Regression. If the method has been successfully completed
  return the elapsed time in seconds.

  @param options - Extra options for the method.
  @return - Elapsed time in seconds or a negative value if the method was not
  successful.
  '''
  def RunTiming(self, options):
    Log.Info("Perform SVR.", self.verbose)

    return self.SVRScikit(options)

