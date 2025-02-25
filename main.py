# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:13:25 2024

@author: jvilp
"""


#%% Import
import numpy as np

from Data import FinancialDataHandler
import observation_data 
from cross_moment_estimator import estimation_matrix_estimator_cross_moment, corrected_matrix_estimator_cross_moment
from non_param_estimation import main_non_param_estimation
from param_estimation import main_param_estimation


#%% Import data
fp = "C:/Users/jvilp/OneDrive/Documents/GestionQuant_volCoJump"
fn = "SP.csv"

df = FinancialDataHandler(fp,fn)
dfilt = df.minutePrice()
print(dfilt.head(10))
Pt = dfilt["price"]


#%% Estimator cross moment 

# Param of cross moment estimator
sigma2_lst = [0.1311, 0.2227,0.3041, 0.3913, 0.4922, 0.6169, 0.7829, 1.0328, 1.5018, 3.0984] # volatility level and their corresponding bandwith
h_list = [0.4165, 0.1811, 0.1374, 0.1173, 0.1126, 0.1131, 0.1239, 0.1496, 0.2134, 0.4397]

# compute of cross moment estimator for various vol level
lst_CM_matrix_estimator = [] 

for i in range(0, len(sigma2_lst)):
    print(i)
    estimated_matrix_cross_moment = estimation_matrix_estimator_cross_moment(np.sqrt(sigma2_lst[i]), h_list[i], Pt)
    corrected_cross_moment = corrected_matrix_estimator_cross_moment(estimated_matrix_cross_moment, 1) #correction in delta=1
    lst_CM_matrix_estimator.append(corrected_cross_moment)


#%% Non parametric estimation

sto_iter = 50 # nb of random init param try
nb_iter_estimation = 50   # nb of iteration of estimation to get confidence interval

df_non_param_component, lst_estimation, lst_implied_cross_moment = main_non_param_estimation(Pt, lst_CM_matrix_estimator, sto_iter, nb_iter_estimation)


#%% Parametric estimation

sto_iter_identity_param = 500 # nb of random init param to compute optimal parameter with W=I
sto_iter_opti_param = 1000 # nb of iteration to found optimal param
nb_simul_var_cov = 600  # nb simul to estimate var-cov matrix
nb_year_var_cov = 20 # histo of 20 year of SP future

eta_opti, matrix_simul_CM = main_param_estimation(sto_iter_identity_param, sto_iter_opti_param, nb_simul_var_cov, nb_year_var_cov, lst_CM_matrix_estimator)


#%% Fait stylisé sur les données

import observation_data

#%% Pricing Kernel

import pricing_kernel

