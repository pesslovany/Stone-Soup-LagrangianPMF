#!/usr/bin/env python

# =====================================
# 4 - Sampling methods: particle filter
# =====================================
# """



# %%
#
# Nearly-constant velocity example
# --------------------------------
# We continue in the same vein as the previous tutorials.
#
# Ground truth
# ^^^^^^^^^^^^
# Import the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import time

from datetime import datetime
from datetime import timedelta


# Initialise Stone Soup ground-truth and transition models.
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.models.transition.linear import KnownTurnRate
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.nonlinear import TerrainAidedNavigation
from stonesoup.models.measurement.linear import LinearGaussian
from scipy.interpolate import RegularGridInterpolator
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import ESSResampler
from stonesoup.resampler.particle import MultinomialResampler
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.functions import gridCreation
from numpy.linalg import inv
from stonesoup.types.state import PointMassState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.types.state import GaussianState

from stonesoup.predictor.pointMass import PointMassPredictor
from stonesoup.updater.pointMass import PointMassUpdater
from scipy.stats import multivariate_normal

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater

from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState
from stonesoup.types.array import StateVectors
import json



# Initialize arrays to store RMSE values
matrixTruePMF = [] 
matrixTruePF = []
matrixTrueKF = []
MC =  10
for mc in range(0,MC):
    print(mc)
    start_time = datetime.now().replace(microsecond=0)
    
    # %%
    
    #np.random.seed(1991)
    
    # %%
    
    
    transition_model = KnownTurnRate(turn_noise_diff_coeffs = [2, 2], turn_rate = np.deg2rad(30))
    
    # This needs to be done in other way
    time_difference = timedelta(days=0, hours=0, minutes=0, seconds=1)
    
    
    timesteps = [start_time]
    truth = GroundTruthPath([GroundTruthState([36569, 50, 55581, 50], timestamp=start_time)])
    
    # %%
    # Create the truth path
    for k in range(1, 20):
        timesteps.append(start_time+timedelta(seconds=k))
        truth.append(GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=timesteps[k]))
    
    
    # %%
    # Initialise the bearing, range sensor using the appropriate measurement model.
    
    # Open the JSON file
    with open('/Users/matoujak/Desktop/file.json', 'r') as file:
        # Load JSON data
        data = json.load(file)
    
    map_x = data['x']
    map_y = data['y']
    map_z = data['z']
    
    map_x = np.array(map_x)
    map_y = np.array(map_y)
    map_z = np.matrix(map_z)
    
    
    interpolator = RegularGridInterpolator((map_x[0,:],map_y[:,0]), map_z)
    
     
  
    measurement_model = TerrainAidedNavigation(interpolator,noise_covar = 1, mapping=(0, 2))
    # matrix = np.array([
    # [1, 0],
    # [0, 1],
    # ])
    # measurement_model = LinearGaussian(ndim_state = 4, mapping = (0, 2), noise_covar = matrix)
    
    # %%
    # Populate the measurement array
    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement, timestamp=state.timestamp,
                                      measurement_model=measurement_model))
    
    
    
    predictor = ParticlePredictor(transition_model)
    resampler = MultinomialResampler()
    updater = ParticleUpdater(measurement_model, resampler)
    
    
    predictorKF = KalmanPredictor(transition_model)
    updaterKF = KalmanUpdater(measurement_model)
    

    
    # %%
    # Initialise a prior
    # ^^^^^^^^^^^^^^^^^^
    # To start we create a prior estimate. This is a :class:`~.ParticleState` which describes
    # the state as a distribution of particles using :class:`~.StateVectors` and weights.
    # This is sampled from the Gaussian distribution (using the same parameters we
    # had in the previous examples).
    
    number_particles = 10000
    
    # Sample from the prior Gaussian distribution
    samples = multivariate_normal.rvs(np.array([36569, 50, 55581, 50]),
                                      np.diag([90, 5, 160, 5]),
                                      size=number_particles)
    
    # Create prior particle state.
    prior = ParticleState(state_vector=StateVectors(samples.T),
                          weight=np.array([Probability(1/number_particles)]*number_particles),
                          timestamp=start_time)
    
    priorKF = GaussianState([[36569], [50], [55581], [50]], np.diag([90, 5, 160, 5]), timestamp=start_time)
    
    # %% PMF prior
    
    pmfPredictor = PointMassPredictor(transition_model)
    pmfUpdater = PointMassUpdater(measurement_model)
    # Initial condition - Gaussian
    nx = 4
    meanX0 = np.array([36569, 50, 55581, 50]) # mean value
    varX0 = np.diag([90, 5, 160, 5]) # variance
    Npa = np.array([31, 31, 27, 27]) # 33 number of points per axis, for FFT must be ODD!!!!
    N = np.prod(Npa) # number of points - total
    sFactor = 4 # scaling factor (number of sigmas covered by the grid)
    
    
    [predGrid, predGridDelta, gridDimOld, xOld, Ppold] = gridCreation(np.vstack(meanX0),varX0,sFactor,nx,Npa)
    
    meanX0 = np.vstack(meanX0)
    pom = predGrid-np.matlib.repmat(meanX0,1,N)
    denominator = np.sqrt((2*np.pi)**nx)*np.linalg.det(varX0)
    pompom = np.sum(-0.5*np.multiply(pom.T@inv(varX0),pom.T),1) #elementwise multiplication
    pomexp = np.exp(pompom)
    predDensityProb = pomexp/denominator # Adding probabilities to points
    predDensityProb = predDensityProb/(sum(predDensityProb)*np.prod(predGridDelta))
    
    priorPMF = PointMassState(state_vector=StateVectors(predGrid),
                          weight=predDensityProb,
                          grid_delta = predGridDelta,
                          grid_dim = gridDimOld,
                          center = xOld,
                          eigVec = Ppold,
                          Npa = Npa,
                          timestamp=start_time)
    
    F = transition_model.matrix(prior=prior, time_interval=time_difference)
    Q = transition_model.covar(time_interval=time_difference)
    
    
    
    priorPMF = PointMassState(state_vector=StateVectors(predGrid),
                          weight=predDensityProb,
                          grid_delta = predGridDelta,
                          grid_dim = gridDimOld,
                          center = xOld,
                          eigVec = Ppold,
                          Npa = Npa,
                          timestamp=start_time)
    
    
    matrixPMF = [] 
       
    start_time = time.time()
    track = Track()
    for measurement in measurements:
        prediction = pmfPredictor.predict(priorPMF, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)
        post = pmfUpdater.update(hypothesis)
        priorPMF = post
        matrixPMF.append(post.mean)
        # print(post.mean)
        
    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time
    # print(end_time - start_time)
     
    
    # matrixKF = [] 
       
    # start_time = time.time()
    # track = Track()
    # for measurement in measurements:
    #     prediction = predictorKF.predict(priorKF, timestamp=measurement.timestamp)
    #     hypothesis = SingleHypothesis(prediction, measurement)
    #     post = updaterKF.update(hypothesis)
    #     priorKF = post
    #     matrixKF.append(post.mean)
    #     # print(post.mean)
        
    # # Record the end time
    # end_time = time.time()
    
    # %%
    # Run the tracker
    # ^^^^^^^^^^^^^^^
    # We now run the predict and update steps, propagating the collection of particles and resampling
    # when told to (at every step).
    
    matrixPF = [] 
    start_time = time.time()
    track = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)
        post = updater.update(hypothesis)
        # print(post.mean)
        track.append(post)
        matrixPF.append(post.mean)
        prior = track[-1]
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time
    # print(end_time - start_time)
    
    
    for ind in range(0,20):
        matrixTruePMF.append(np.ravel(np.vstack(matrixPMF[ind])-truth.states[ind].state_vector))
        matrixTruePF.append(np.ravel(matrixPF[ind]-truth.states[ind].state_vector)) 
        # matrixTrueKF.append(np.ravel(matrixKF[ind]-truth.states[ind].state_vector)) 
        
    
def rmse(errors):
    """
    Calculate the Root Mean Square Error (RMSE) from a list of errors.

    Args:
        errors (list): List of errors.

    Returns:
        float: RMSE value.
    """
    # Convert the list of errors into a numpy array for easier computation
    errors_array = np.array(errors)

    # Square the errors
    squared_errors = np.square(errors_array)

    # Calculate the mean squared error
    mean_squared_error = np.mean(squared_errors,0)

    # Calculate the root mean squared error
    rmse_value = np.sqrt(mean_squared_error)

    return rmse_value
    
    
print(rmse(matrixTruePF))
print(rmse(matrixTruePMF)) 
# print(rmse(matrixTrueKF))




