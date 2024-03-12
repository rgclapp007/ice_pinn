"""Import external packages"""
import sys
import time
from pathlib import Path
import math as m
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import matplotlib.pyplot as plt
import os

"""Import helper functions for synthetic data generation, initializing neural networks, initializing losses, etc."""
from data.noise import add_noise
from data.sample import random_sample
from model import create_mlp
from loss import SquareLoss, SquareLossRandom
#from loss_colo import SquareLossRandom
from optimization import LBFGS, Adam
from formulations.constants import *
from formulations.helpers import get_collocation_points, to_mat_tensor, to_tensor
from formulations.helpers import _data_type

"""Indexing for parallelizing multiple trials at fixed noise + one or multiple gamma values using SLURM. Comment out if running an individual trial without SLURM."""
#idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
#expnum = str(idx) #experiment number label for result files

"""Add location of ground truth data to sys.path"""
sys.path.append("/home/yiwasaki/IceShelf1D")

"""Determine which equations to use"""
from formulations.eqns_o1_inverse import Data_Equations, Inverse_1stOrder_Equations

"""Define the domain of the problem"""
#Import ground truth data for u,h and their x-positions (x) from which to build synthetic noisy training data
data = loadmat('constantB_uh.mat') #file path to ground truth u(x), h(x) profiles. To test sinusoidal B(x) studied in our paper, replace with location of "sinusoidalB_uh.mat".
x_star = np.transpose(data['x']) 
u_star = np.transpose(data['u'])[:, 0]
h_star = np.transpose(data['h'])[:, 0]
B_truth = np.ones_like(x_star) #B(x) profile used to solve for ground truth u and h profiles. REPLACE rhs with 0.5*np.cos(3*np.pi*x_star) + 1 to test the sinusoidal profile studied in our paper.

"""Parameters"""
# Data parameters
N_t = 1001  # Number of collocation points
N_ob = 401  # Number of training points.

# Model parameters
layers = [20,20,20,20,20,20, 3] #Number of hidden units in each layer.
lyscl = [1, 1, 1, 1, 1, 1] #Standard deviation to set the scales for Xavier weight initialization.

# Hyper parameters for the PINN
fractional = False

num_iterations_adam_resampled = 4000   #Number of iterations of Adam using collocation resampling
num_iterations_adam_fixed = 2000      #Number of iterations of Adam with fixed collocation points
num_iterations_lbfgs = 2000           #umber of iterations of LBFGS using fixed collocation points

"""Function to generate synthetic training data at a specified noise level, then train PINNs at a fixed value of gamma."""
def Berr_func(gamma, noise_level, x_star, u_star, h_star, layers):
    
    #Arguments:
    #gamma:       value of gamma used for training
    #noise_level: level of noise added to synthetic training data
    #x_star:      x-locations of ground truth u and h training data
    #u_star:      ground truth u data at x-locations given by x_star
    #h_star:      ground truth h data at x-locations given by x_star
    
    model = create_mlp(layers, lyscl, dtype=_data_type) #initialize neural network model
    equations = Inverse_1stOrder_Equations(fractional=fractional) #set governing physics equations (1D SSA)

    # generate synthetic training data by randomly sampling ground truth data points and adding noise at desired noise level
    x_sampled, u_sampled, h_sampled = random_sample(
        N_ob, x_star,
        add_noise(u_star, ratio=noise_level),
        add_noise(h_star, ratio=noise_level)
    )

    loss_colo       = SquareLossRandom(equations=equations, equations_data=Data_Equations, gamma=gamma) #Initialize loss function for collocation resampled training

    collocation_pts = get_collocation_points(x_train=x_star, xmin=x_star.min(), xmax=x_star.max(), N_t=N_t) #Randomly sample a set of collocation points to be used for fixed collocation training
    loss            = SquareLoss(equations=equations, equations_data=Data_Equations, gamma=gamma) #Initialize loss function for fixed collocation training

    
    # Record training time
    start_time = time.time()

    #Train using Adam with collocation resampling: initialize Adam optimizer with argument loss=loss_colo to enable collocation resampling.
    # NOTE: the Adam() initializer requires a "collocation_points" argument even when using collocation resampling. This was done for consistency with the fixed collocation version.
    # Whatever points are passed as argument are ignored when loss_colo is passed as the loss argument; in this case an empty array of collocation points can also be passed without error.
    
    adam_resampled = Adam(
        net=model, loss=loss_colo, collocation_points=collocation_pts,
        data_points=(to_tensor(x_sampled), (to_tensor(u_sampled), to_tensor(h_sampled)))
    )
    adam_resampled.optimize(nIter=num_iterations_adam_resampled)

    #Train using Adam with fixed collocation points: initialize Adam optimizer with argument loss=loss for fixed collocation points.
    adam_fixed = Adam(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(x_sampled), (to_tensor(u_sampled), to_tensor(h_sampled)))
    )
    adam_fixed.optimize(nIter=num_iterations_adam_fixed)

    #Train using LBFGS with fixed collocation points
    lbfgs = LBFGS(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(x_sampled), (to_tensor(u_sampled), to_tensor(h_sampled)))
    )
    lbfgs.optimize(nIter=num_iterations_lbfgs)
    elapsed = time.time() - start_time

    #Report total training time in seconds
    print('Training time: %.4f' % elapsed)

    #Combine and process equation, data, and total losses from each training session (i.e. for training completed with each optimizer and collocation method)
    equation_losses = np.array(adam_resampled.loss_records.get("loss_equation", [])+ (adam_fixed.loss_records.get("loss_equation", []) + lbfgs.loss_records.get("loss_equation", []))
    data_losses = np.array(adam_resampled.loss_records.get("loss_data", []) + adam_fixed.loss_records.get("loss_data", []) + lbfgs.loss_records.get("loss_data", []))
    total_losses = np.array(adam_resampled.loss_records.get("loss", []) + adam_fixed.loss_records.get("loss", []) + lbfgs.loss_records.get("loss", []))

    data_losses = np.trim_zeros(data_losses, 'b')
    equation_losses = np.trim_zeros(equation_losses, 'b')
    total_losses = np.trim_zeros(total_losses, 'b')

    
    ############################# Data Analysis + Processing ###############################

    x_star = tf.cast(x_star, dtype=_data_type)
    uhb_pred = model(x_star)
    f_pred = equations(x_star, model, drop_mass_balance=False)
    x_star = x_star.numpy().flatten().flatten()
    x_sampled = x_sampled.flatten().flatten()
    u_star = u_star.flatten()
    u_sampled = u_sampled.flatten()
    h_star = h_star.flatten()
    h_sampled = h_sampled.flatten()
    u_p = uhb_pred[:, 0:1].numpy().flatten()
    h_p = uhb_pred[:, 1:2].numpy().flatten()
    B_p = uhb_pred[:, 2:3].numpy().flatten()

    #compute B_err, u_err, h_err
    total_berr = err(B_p, B_truth)
    total_uerr = err(u_p, u_star)
    total_herr = err(h_p, h_star)

    #return a dictionary of results
    results = { "B_err" : total_berr,
                "u_err" : total_uerr,
                "h_err" : total_herr,
                "B_p" : B_p, 
                "data_losses" : data_losses, 
                "equation_losses" : equation_losses, 
                "total_losses" : total_losses, 
                "u_p" : u_p,
                "h_p" : h_p,
                "u_sampled" : u_sampled,
                "h_sampled" : h_sampled
               }   
    return results

#evaluate B_err/u_err/h_err.
def err(B_p, B_truth):
#Arguments:
           #B_p: final profile for a given variable predicted by NN
           #B_truth: ground truth profile of the same variable
    N = B_p.size
    return (1/N)*np.sum(np.square(B_p-B_truth))
           
# helper function to test and return the results for different values of gamma after calling one script
def gamma_batch(test_gammas, noise_level, x_star, u_star, h_star, layers): 
    batch_results = []
    for i, gamma in enumerate(test_gammas):
        #store result of training into dictionary
        exp_dic = Berr_func(gamma, noise_level, x_star, u_star, h_star, layers)
        batch_results.append(exp_dic)
    return batch_results

#helper function for storing the results from one or more values of gamma in one call. 
def format_dict(dict_list):
    berrs = []
    uerrs = []
    herrs = []
    bpreds = []
    d_losses = []
    e_losses = []
    t_losses = []
    u_preds = []
    h_preds = []
    u_samp = []
    h_samp = []

    for i in range(len(dict_list)):
        berrs.append(dict_list[i]["B_err"])
        uerrs.append(dict_list[i]["u_err"])
        herrs.append(dict_list[i]["h_err"])
        bpreds.append(dict_list[i]["B_p"])
        d_losses.append(dict_list[i]["data_losses"])
        e_losses.append(dict_list[i]["equation_losses"])
        t_losses.append(dict_list[i]["total_losses"])
        u_preds.append(dict_list[i]["u_p"])
        h_preds.append(dict_list[i]["h_p"])
        u_samp.append(dict_list[i]["u_sampled"])
        h_samp.append(dict_list[i]["h_sampled"])

    new_dict = {"berrs" : np.asarray(berrs),
                "uerrs" : np.asarray(uerrs),
                "herrs" : np.asarray(herrs),
                "bpreds" : np.asarray(bpreds),
                "d_losses" : d_losses, #just keep as list
                "e_losses" : e_losses, #just keep as list
                "t_losses" : t_losses, #just keep as list
                "u_p" : np.asarray(u_preds),
                "h_p" : np.asarray(h_preds),
                "u_sampled" : np.asarray(u_samp),
                "h_sampled" : np.asarray(h_samp)
               }
    return new_dict
    
# set the noise value of the training data
test_noise = 0.3

#select gammas to test: choose gamma ratios logarithmically from 10^-4 to 10^8
logratios = np.linspace(-4,8,13)
test_gammas = np.power(10,logratios)/(1+np.power(10,logratios))

#test several values of gamma and store results in a python dictionary
results = gamma_batch(test_gammas, test_noise, x_star, u_star, h_star, layers) #test a range of gammas for noise = 0.3
result_dict    = format_dict(results)

from scipy.io import savemat
file_str = "r" #+ expnum #label results by the experiment trial number. Comment out "expnum" if running a single trial without SLURM.
savemat('/home/pinntrial_results/' + file_str + '.mat', result_dict) #save trial results to folder **MODIFY TO DESIRED RESULTS DIRECTORY**

