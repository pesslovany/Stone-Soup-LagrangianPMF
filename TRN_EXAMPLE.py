# %%
# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time

from datetime import datetime
from datetime import timedelta
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Initialise Stone Soup ground-truth and transition models.
from stonesoup.models.transition.linear import KnownTurnRate, ConstantVelocity, CombinedLinearGaussianTransitionModel
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.nonlinear import TerrainAidedNavigation
from scipy.interpolate import RegularGridInterpolator
from stonesoup.predictor.particle import ParticlePredictor

from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater

from stonesoup.resampler.particle import SystematicResampler
from stonesoup.resampler.particle import ResidualResampler
from stonesoup.resampler.particle import StratifiedResampler
from stonesoup.resampler.particle import ESSResampler

from stonesoup.types.state import GaussianState

from stonesoup.updater.particle import ParticleUpdater
from stonesoup.functions import grid_creation
from numpy.linalg import inv
from stonesoup.types.state import PointMassState
from stonesoup.types.hypothesis import SingleHypothesis

from stonesoup.predictor.pointmass import PointMassPredictor
from stonesoup.updater.pointmass import PointMassUpdater
from scipy.stats import multivariate_normal
from scipy.io import loadmat

from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState
from stonesoup.types.array import StateVectors

from stonesoup.plotter import AnimatedPlotterly
import plotly.io as pio
pio.renderers.default = "vscode"

######################## Dynamics Setup ########################
# Define starting position
vRanges               = [(-70, -50), (50, 70)]
trRanges              = [(-np.deg2rad(0.2), -np.deg2rad(0.1)), (np.deg2rad(0.1), np.deg2rad(0.2))]
r0                    = np.array([60000,35000])
P0                    = np.diag([120,20,120,20])
# Define turn rate ranges (excluding values near 0)
deltaT                = 3
nTime                 = 150
Qn                    = 1e-3
nS                    = 4
MC                    = 1

######################## Measurement Setup ########################
# Import map
data                  = loadmat(r'MapTAN.mat')
map_x                 = np.array(data['map_m'][0][0][0])
map_y                 = np.array(data['map_m'][0][0][1])
map_z                 = np.matrix(data['map_m'][0][0][2])
interpolator          = RegularGridInterpolator((map_x[:,0],map_y[0,:]),map_z)
Rmap                  = 0.5
measurement_model     = TerrainAidedNavigation(interpolator,noise_covar = Rmap, mapping = (0, 2))

######################## Initialize Arrays ########################
# Initialize result variables for PMF+GSF
errorGMF              = np.zeros(shape = (nS, nTime, MC))
stateGMF              = np.zeros(shape = (nS, nTime, MC))
covGMF                = np.zeros(shape = (nS, nS, nTime, MC))
neesGMF               = np.zeros(shape = (1, nTime, MC))
# Initialize result variables for PMF
errorPMF              = np.zeros(shape = (nS, nTime, MC)) 
statePMF              = np.zeros(shape = (nS, nTime, MC))
covPMF                = np.zeros(shape = (nS, nS, nTime, MC))
neesPMF               = np.zeros(shape = (1, nTime, MC))
# Initialize result variables for UKF
errorUKF              = np.zeros(shape = (nS, nTime, MC)) 
stateUKF              = np.zeros(shape = (nS, nTime, MC))
covUKF                = np.zeros(shape = (nS, nS, nTime, MC))
neesUKF               = np.zeros(shape = (1, nTime, MC))
# Initialize result variables for Systematic Particle Filter
errorPFS              = np.zeros((nS, nTime, MC))
statePFS              = np.zeros((nS, nTime, MC))
covPFS                = np.zeros((nS, nS, nTime, MC))
neesPFS               = np.zeros((1, nTime, MC))
# Preallocate the end timing arrays for each method
end_time_GMF          = np.zeros(MC)
end_time_PMF          = np.zeros(MC)
end_time_PFS          = np.zeros(MC)
end_time_UKF          = np.zeros(MC)
# Preallocate the start timing arrays for each method
start_time_GMF        = np.zeros(MC)
start_time_PMF        = np.zeros(MC)
start_time_PFS        = np.zeros(MC)
start_time_UKF        = np.zeros(MC)

######################## Monte Carlo Runs ########################
for mc in range(0,MC):
    
    ######################## MC Settings ########################
    # Set seed
    np.random.seed(8)
    print(mc)
    
    ############################ Run truth trajectory ########################
    # True state
    vRange              = vRanges[np.random.choice([0, 1])]
    v0                  = np.random.uniform(vRange[0], vRange[1])
    trRange             = trRanges[np.random.choice([0, 1])]
    turnRate            = np.random.uniform(trRange[0], trRange[1])
    X0                  = np.array([r0[0],v0,r0[1],0])
    # Truth settings
    start_time          = datetime.now().replace(microsecond=0)
    transition_model    = KnownTurnRate(turn_noise_diff_coeffs = [Qn,Qn], turn_rate = turnRate)
    timesteps           = [start_time]
    truth               = GroundTruthPath([GroundTruthState(np.random.multivariate_normal(X0,P0), timestamp = start_time)])
    # Create the truth path
    plt.figure()
    for k in range(1,nTime):
        timesteps.append(start_time + deltaT*timedelta(seconds = k))
        truth.append(GroundTruthState(transition_model.function(truth[k - 1], noise = True, time_interval = timedelta(seconds = deltaT)),timestamp = timesteps[k]))
        plt.scatter(truth[k].state_vector[0],truth[k].state_vector[2],s = 50, color = 'blue',marker = 'o',alpha = 0.1)

    # Populate the measurement array
    measurements        = []
    for state in truth:
        measurement = measurement_model.function(state, noise = True)
        measurements.append(Detection(measurement, timestamp = state.timestamp, measurement_model = measurement_model))

    ############################ Run Filters ############################
    # Initialize UKF
    predictorUKF        = UnscentedKalmanPredictor(transition_model)
    updaterUKF          = UnscentedKalmanUpdater(measurement_model)
    postUKF             = GaussianState(X0, P0, timestamp=start_time)
    # Run UKF
    start_time_UKF[mc]  = time.time()
    kTime               = 0
    for measurement in measurements:
        prediction           = predictorUKF.predict(postUKF, timestamp = measurement.timestamp)
        hypothesis           = SingleHypothesis(prediction, measurement)
        postUKF              = updaterUKF.update(hypothesis)
        plt.scatter(postUKF.mean[0],postUKF.mean[2],s = 2,color = 'black',marker = 'o',alpha = 1)
        if np.isnan(postUKF.mean[0]):
            print('UKF')
            errorUKF[:,kTime:-1,mc] = 1e3
            stateUKF[:,kTime:-1,mc] = 1e3
            covUKF[:,:,kTime:-1,mc] = 1e3
            neesUKF[:,kTime:-1,mc]  = 1e3
            break
        errorUKF[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - postUKF.mean.T
        stateUKF[:,kTime,mc] = postUKF.mean.T
        covUKF[:,:,kTime,mc] = np.matrix(postUKF.covar)
        neesUKF[:,kTime,mc]  = errorUKF[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covUKF[:,:,kTime,mc]) @ errorUKF[:,kTime,mc].reshape(nS,1)
        kTime               += 1
    end_time_UKF[mc] = time.time()

    del prediction, hypothesis, postUKF

    # Initialize PMF+GSF
    predictorGMF        = PointMassPredictor(transition_model)
    updaterGMF          = PointMassUpdater(measurement_model)
    Npa                 = np.array([9, 9, 9, 9])    # for FFT must be ODD!!!!
    N                   = np.prod(Npa)              # number of points - total
    sFactor             = 6                         # scaling factor (number of sigmas covered by the grid)
    [predGrid, predGridDelta, gridDimOld, xOld, Ppold] = grid_creation(np.vstack(X0),P0,sFactor,nS,Npa)
    meanX0              = np.vstack(X0)
    pom                 = predGrid - np.tile(meanX0, (1, N))
    denominator         = np.sqrt((2*np.pi)**nS)*np.linalg.det(P0)
    pompom              = np.sum(-0.5*np.multiply(pom.T@inv(P0),pom.T),1) # elementwise multiplication
    pomexp              = np.exp(pompom)
    predDensityProb     = pomexp/denominator # Adding probabilities to points
    predDensityProb     = predDensityProb/(sum(predDensityProb)*np.prod(predGridDelta))
    postGMF             = PointMassState(state_vector = StateVectors(predGrid),
                                          weight       = predDensityProb,
                                          grid_delta   = predGridDelta,
                                          grid_dim     = gridDimOld,
                                          center       = xOld,
                                          eigVec       = Ppold,
                                          Npa          = Npa,
                                          timestamp    = start_time)
    
    # Run PMF+GSF
    start_time_GMF[mc]  = time.time()
    kTime               = 0
    for measurement in measurements:
        prediction           = predictorGMF.predict(postGMF, timestamp = measurement.timestamp, runGSFversion = True, futureMeas = measurement, measModel = measurement_model)
        hypothesis           = SingleHypothesis(prediction, measurement)
        postGMF              = updaterGMF.update(hypothesis)
        plt.scatter(postGMF.mean[0],postGMF.mean[2],s = 2,color = 'red',marker = 'o',alpha = 1)
        if np.isnan(postGMF.mean[0]):
            print('GMF')
            errorGMF[:,kTime:-1,mc] = 1e3
            stateGMF[:,kTime:-1,mc] = 1e3
            covGMF[:,:,kTime:-1,mc] = 1e3
            neesGMF[:,kTime:-1,mc]  = 1e3
            break
        errorGMF[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - postGMF.mean
        stateGMF[:,kTime,mc] = postGMF.mean
        covGMF[:,:,kTime,mc] = np.matrix(postGMF.covar())
        neesGMF[:,kTime,mc]  = errorGMF[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covGMF[:,:,kTime,mc]) @ errorGMF[:,kTime,mc].reshape(nS,1)
        kTime               += 1
    end_time_GMF[mc]    = time.time()

    del prediction, hypothesis, postGMF, predDensityProb, predGrid

    # Initialize PMF
    predictorPMF        = PointMassPredictor(transition_model)
    updaterPMF          = PointMassUpdater(measurement_model)
    Npa                 = np.array([9, 9, 9, 9])    # for FFT must be ODD!!!!
    N                   = np.prod(Npa)              # number of points - total
    sFactor             = 6                         # scaling factor (number of sigmas covered by the grid)
    [predGrid, predGridDelta, gridDimOld, xOld, Ppold] = grid_creation(np.vstack(X0),P0,sFactor,nS,Npa)
    meanX0              = np.vstack(X0)
    pom                 = predGrid - np.tile(meanX0, (1, N))
    denominator         = np.sqrt((2*np.pi)**nS)*np.linalg.det(P0)
    pompom              = np.sum(-0.5*np.multiply(pom.T@inv(P0),pom.T),1) #elementwise multiplication
    pomexp              = np.exp(pompom)
    predDensityProb     = pomexp/denominator # Adding probabilities to points
    predDensityProb     = predDensityProb/(sum(predDensityProb)*np.prod(predGridDelta))
    postPMF             = PointMassState(state_vector = StateVectors(predGrid),
                                         weight       = predDensityProb,
                                         grid_delta   = predGridDelta,
                                         grid_dim     = gridDimOld,
                                         center       = xOld,
                                         eigVec       = Ppold,
                                         Npa          = Npa,
                                         timestamp    = start_time)
    
    # Run PMF
    start_time_PMF[mc]  = time.time()
    kTime               = 0
    for measurement in measurements:
        prediction           = predictorPMF.predict(postPMF, timestamp = measurement.timestamp, runGSFversion = False, futureMeas = measurement, measModel = measurement_model)
        hypothesis           = SingleHypothesis(prediction, measurement)
        postPMF              = updaterPMF.update(hypothesis)
        plt.scatter(postPMF.mean[0],postPMF.mean[2],s = 2,color = 'cyan',marker = 'o',alpha = 1)
        if np.isnan(postPMF.mean[0]):
            print('PMF')
            errorPMF[:,kTime:-1,mc] = 1e3
            statePMF[:,kTime:-1,mc] = 1e3
            covPMF[:,:,kTime:-1,mc] = 1e3
            neesPMF[:,kTime:-1,mc]  = 1e3
            break
        errorPMF[:,kTime,mc] = np.array(truth.states[kTime].state_vector).T - postPMF.mean
        statePMF[:,kTime,mc] = postPMF.mean
        covPMF[:,:,kTime,mc] = np.matrix(postPMF.covar())
        neesPMF[:,kTime,mc]  = errorPMF[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covPMF[:,:,kTime,mc]) @ errorPMF[:,kTime,mc].reshape(nS,1)
        kTime               += 1
    end_time_PMF[mc]    = time.time()

    del prediction, hypothesis, postPMF, predDensityProb, predGrid
    
    # Initialize PF-Systematic
    nParticlesS                 = np.round(2.5*N).astype(int)
    predictorPFS                = ParticlePredictor(transition_model)
    resamplerPFS                = SystematicResampler()
    resamplerPFS                = ESSResampler(threshold = nParticlesS*0.8, resampler = resamplerPFS)
    updaterPFS                  = ParticleUpdater(measurement_model, resamplerPFS)
    samplesPFS                  = multivariate_normal.rvs(X0, P0, size = nParticlesS)
    postPFS                     = ParticleState(state_vector = StateVectors(samplesPFS.T), weight = np.array([Probability(1/nParticlesS)]*nParticlesS), timestamp = start_time)

    # Run PF-Systematic
    start_time_PFS[mc]          = time.time()
    kTime                       = 0
    for measurement in measurements:
        prediction                      = predictorPFS.predict(postPFS, timestamp = measurement.timestamp)
        hypothesis                      = SingleHypothesis(prediction, measurement)
        postPFS                         = updaterPFS.update(hypothesis)
        plt.scatter(postPFS.mean[0],postPFS.mean[2],s = 2,color = 'green',marker = 'o',alpha = 1)
        if np.isnan(postPFS.mean[0]):
            print('PFS')
            errorPFS[:,kTime:-1,mc] = 1e3
            statePFS[:,kTime:-1,mc] = 1e3
            covPFS[:,:,kTime:-1,mc] = 1e3
            neesPFS[:,kTime:-1,mc]  = 1e3
            break
        priorPFS              = postPFS
        errorPFS[:,kTime,mc]  = np.array(truth.states[kTime].state_vector).T - np.array(postPFS.mean).T
        statePFS[:,kTime,mc]  = np.array(postPFS.mean).T
        covPFS[:,:,kTime,mc]  = np.matrix(postPFS.covar)
        neesPFS[:,kTime,mc]   = errorPFS[:,kTime,mc].reshape(1,nS) @ np.linalg.pinv(covPFS[:,:,kTime,mc]) @ errorPFS[:,kTime,mc].reshape(nS,1)
        kTime                           += 1
    end_time_PFS[mc]            = time.time()
    del prediction, hypothesis, postPFS, priorPFS, samplesPFS

plt.show()

############################ Plotting ############################
# Plotting Settings
plt.rc('font', family='serif', serif=['Computer Modern'])
plt.rc('text', usetex=True)
plt.rcParams['axes.linewidth']   = 2 # Thicker axes
plt.rcParams['lines.linewidth']  = 2 # Thicker lines
plt.rcParams['xtick.major.size'] = 7 # Major tick length
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = 4 # Minor tick length
plt.rcParams['ytick.minor.size'] = 4
x_vals                           = np.linspace(1,nTime,nTime)
translucent_blue                 = (0/255,114/255,178/255,0.3)
y_labels                         = [r'$\tilde{r}_x$ (m)',r'$\tilde{v}_x$ (m/s)',r'$\tilde{r}_y$ (m)',r'$\tilde{v}_y$ (m/s)']
cb_colors = {
    'neutral': '#000000',  # black/neutral
    'mean':    '#000000',  # black
    'upper':   '#D55E00',  # red
    'lower':   '#D55E00',  # same for lower bound
    'std':     '#0072B2'   # blue
}
# Boxplot for RMSE and SNEES comparison
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# RMSE Position
data_1 = np.mean(np.sqrt(np.mean(errorGMF[[0, 2], :, :]**2, axis=0)), axis=0)
print(np.mean(data_1))
data_2 = np.mean(np.sqrt(np.mean(errorPMF[[0, 2], :, :]**2, axis=0)), axis=0)
print(np.mean(data_2))
data_3 = np.mean(np.sqrt(np.mean(errorPFS[[0, 2], :, :]**2, axis=0)), axis=0)
print(np.mean(data_3))
data_4 = np.mean(np.sqrt(np.mean(errorUKF[[0, 2], :, :]**2, axis=0)), axis=0)
print(np.mean(data_4))
data   = [data_1, data_2, data_3, data_4]
bp     =  axs[0].boxplot(data,patch_artist = True,
                    boxprops = dict(facecolor = translucent_blue,color = cb_colors['neutral'],linewidth = 2),
                    medianprops = dict(color = cb_colors['upper'],linewidth = 2))
axs[0].minorticks_on()
axs[0].tick_params(which='both', width=2)
axs[0].tick_params(which='major', length=7)
axs[0].tick_params(which='minor', length=4)
axs[0].set_xticks([1, 2, 3, 4])
axs[0].set_xticklabels(['GMF', 'PMF', 'PFS', 'UKF'])
axs[0].set_ylabel(r'\textbf{RMSE} Position (m)')
# Create inset for RMSE Position
axins0 = inset_axes(axs[0], width="50%", height="50%", loc='upper left', borderpad=2)
bp_ins = axins0.boxplot(data, patch_artist=True,
                        boxprops=dict(facecolor=translucent_blue, color=cb_colors['neutral'], linewidth=2),
                        medianprops=dict(color=cb_colors['upper'], linewidth=2))
axins0.set_ylim([4, 10])
axins0.set_xticks([])
axins0.yaxis.tick_right() 
# RMSE Velocity
data_1 = np.mean(np.sqrt(np.mean(errorGMF[[1, 3], :, :]**2, axis=0)), axis=0)
data_2 = np.mean(np.sqrt(np.mean(errorPMF[[1, 3], :, :]**2, axis=0)), axis=0)
data_3 = np.mean(np.sqrt(np.mean(errorPFS[[1, 3], :, :]**2, axis=0)), axis=0)
data_4 = np.mean(np.sqrt(np.mean(errorUKF[[1, 3], :, :]**2, axis=0)), axis=0)
data   = [data_1, data_2, data_3, data_4]
bp     =  axs[1].boxplot(data,patch_artist = True,
                    boxprops = dict(facecolor = translucent_blue,color = cb_colors['neutral'],linewidth = 2),
                    medianprops = dict(color = cb_colors['upper'],linewidth = 2))
axs[1].minorticks_on()
axs[1].tick_params(which='both', width=2)
axs[1].tick_params(which='major', length=7)
axs[1].tick_params(which='minor', length=4)
axs[1].set_xticks([1, 2, 3, 4])
axs[1].set_xticklabels(['GMF', 'PMF', 'PFS', 'UKF'])
axs[1].set_ylabel(r'\textbf{RMSE} Velocity (m/s)')
axins1 = inset_axes(axs[1], width="50%", height="50%", loc='upper left', borderpad=2)
bp_ins = axins1.boxplot(data, patch_artist=True,
                        boxprops=dict(facecolor=translucent_blue, color=cb_colors['neutral'], linewidth=2),
                        medianprops=dict(color=cb_colors['upper'], linewidth=2))
axins1.set_ylim([0, 1])
axins1.set_xticks([])
axins1.yaxis.tick_right() 
# SNEES
data_1 = np.median(neesGMF,axis = 1)[0]/nS
data_2 = np.median(neesPMF,axis = 1)[0]/nS
data_3 = np.median(neesPFS,axis = 1)[0]/nS
data_4 = np.median(neesUKF,axis = 1)[0]/nS
data   = [data_1, data_2, data_3, data_4]
bp     =  axs[2].boxplot(data,patch_artist = True,
                    boxprops = dict(facecolor = translucent_blue,color = cb_colors['neutral'],linewidth = 2),
                    medianprops = dict(color = cb_colors['upper'],linewidth = 2))
axs[2].minorticks_on()
axs[2].tick_params(which='both', width=2)
axs[2].tick_params(which='major', length=7)
axs[2].tick_params(which='minor', length=4)
axs[2].set_xticks([1, 2, 3, 4])
axs[2].set_xticklabels(['GMF', 'PMF', 'PFS', 'UKF'])
axs[2].set_ylabel(r'\textbf{SNEES} Position')
axins2 = inset_axes(axs[2], width="50%", height="50%", loc='upper left', borderpad=2)
bp_ins = axins2.boxplot(data, patch_artist=True,
                        boxprops=dict(facecolor=translucent_blue, color=cb_colors['neutral'], linewidth=2),
                        medianprops=dict(color=cb_colors['upper'], linewidth=2))
axins2.set_ylim([0, 5])
axins2.set_xticks([])
axins2.yaxis.tick_right() 

############################ Time ############################
# Calculate the mean times for each filter method
mean_time_GMF           = np.mean(end_time_GMF - start_time_GMF)/nTime
mean_time_PMF           = np.mean(end_time_PMF - start_time_PMF)/nTime
mean_time_PFS           = np.mean(end_time_PFS - start_time_PFS)/nTime
mean_time_UKF           = np.mean(end_time_UKF - start_time_UKF)/nTime
# Create a list of the mean times and filter labels
mean_times              = [mean_time_GMF, mean_time_PMF, mean_time_PFS, mean_time_UKF]
filters                 = ['GMF', 'PMF', 'PFS', 'UKF']
# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(filters, mean_times, color='skyblue', edgecolor='black')
# Set labels and title
ax.set_xlabel('Filter Method', fontsize=14)
ax.set_ylabel('Mean Time (s)', fontsize=14)
ax.set_title('Mean Time for Each Filter Method', fontsize=16)

# Show plots
plt.show()