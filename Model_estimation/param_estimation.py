# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:41:16 2024

@author: jvilp
"""

#error handling so as not to stop the code with large simulation
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
from numpy.linalg import LinAlgError
from contextlib import contextmanager

@contextmanager
def catch_complex_warning():
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", np.ComplexWarning)  
        yield captured_warnings


import numpy as np
from scipy.optimize import minimize
from cross_moment_estimation import cross_moment
from simulation_parametric import simulation_full_model
from cross_moment_estimator import estimation_matrix_estimator_cross_moment, corrected_matrix_estimator_cross_moment





def cost_function_param(eta, W, sigma2_lst, lst_CM_matrix_estimator, used_moment):
    mu_r = eta[0]
    rho_0 = eta[1]
    rho_1 = eta[2]
    m0 = eta[3]
    m1 = eta[4]
    delta = eta[5]
    mu_Jr = eta[6]
    mu_JJr0 = eta[7]
    mu_JJr1 = eta[8]
    sigma_Jr = eta[9]
    sigma_JJr0 = eta[10]
    sigma_JJr1 = eta[11]
    mu_Jsigma = eta[12]
    mu_JJsigma = eta[13]
    sigma_Jsigma = eta[14]
    sigma_JJsigma = eta[15]
    rho_J = eta[16]
    lambda_r = eta[17]
    lambda_sigma = eta[18]
    lambda_r_sigma = eta[19]
    
    S = len(used_moment)
    G = len(sigma2_lst)
    
    CM_true = np.zeros((S*G,1))
    CM_estim = np.zeros((S*G,1))
    
    for g in range(0,G):
        for s in range(0,S):
            p1 = used_moment[s,0]
            p2 = used_moment[s,1]
            CM_estim[g*S + s,0] = lst_CM_matrix_estimator[g][p1,p2]
            
            sigma = np.sqrt(sigma2_lst[g])
            rho = max(min(rho_0 + rho_1*sigma,1),-1)
            m = m0+m1*np.log(sigma**2)
            mu_JJr = mu_JJr0 + mu_JJr1 * sigma
            
            sigma_JJr = sigma_JJr0 + sigma_JJr1 * sigma
            CM_true[g*S + s,0] = cross_moment(p1, p2, sigma, mu_r, rho, m, delta, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)


    bias = CM_estim - CM_true
    cost_function = np.dot(bias.T,np.dot(W,bias))
    
    return cost_function[0,0]
    
    
    
def parametric_estimation(W, sigma2_lst, lst_CM_matrix_estimator, used_moment, sto_iter, eta_init = None):
    bounds = [(None, None), # mu_r 
              (None, None), # rho_0
              (None, None), # rho_1
              (None, None), # m_0
              (None, None), # m_1
              (0, None),    # delta
              (None, None), # mu_Jr
              (None, None), # mu_JJr0
              (None, None), # mu_JJr1
              (0, None),    # sigma_Jr
              (None, None), # sigma_JJ0
              (None, None), # sigma_JJ1
              (None, None), # mu_Jsigma
              (None, None), # mu_JJsigma
              (0, None),    # sigma_Jsigma
              (0, None),    # sigma_JJsigma
              (-1, 1),      # rho_J
              (0, None),    # lambda_r
              (0, None),    # lamda_sigma
              (0, None)]    # lambda_r_sigma
    
    sigma_max = np.sqrt(sigma2_lst[-1])  # we define constrain on parameter to ensure rho and sigma_JJ composant be well represented
    sigma_min = np.sqrt(sigma2_lst[0])
    
    constraints = (
    {'type': 'ineq', 'fun': lambda x: x[1] + x[2]*sigma_max + 1},  # rho_0 + rho_1 * sigma_max > -1  as cond > 0 with type = "ineq"
    {'type': 'ineq', 'fun': lambda x: x[1] + x[2]*sigma_min + 1},  # rho_0 + rho_1 * sigma_min > -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[1] - x[2]*sigma_max},  # rho_0 + rho_1 * sigma_max < 1
    {'type': 'ineq', 'fun': lambda x: 1 - x[1] - x[2]*sigma_min},  # rho_0 + rho_1 * sigma_min < 1
    
    {'type': 'ineq', 'fun': lambda x: x[10] + x[11] * sigma_max},  # we ensure that for all sigma value sigma_JJ0 + sigma_JJ1 * sigma > 0
    {'type': 'ineq', 'fun': lambda x: x[10] + x[11] * sigma_min}) 
    
    
    if eta_init is None:
        min_fun = 10**20
        for r in range(0, sto_iter):           
            print("iter = ",r)
            try_compute = True
            while try_compute:  # we catch error in case of random init parameter with identity make error 
                try:
                    with catch_complex_warning() as captured_warnings:
                        
                        eta_init = np.random.normal(0, 0.25, 20) # random init for the 20 parameter normal law N(0,0.25) (fixed len=20 as fix number of component in our parametric sde system)
            
                        # we verifie uur stochastic init param verify proper condition constrain
                        while any(constraints[i]["fun"](eta_init) < 0 for i in range(0, 4)):
                            eta_init[1], eta_init[2] = np.random.normal(0, 0.25, 2)
                            
                        while any(constraints[i]["fun"](eta_init) < 0 for i in range(4, 6)):
                            eta_init[10], eta_init[11], eta_init[12] = np.random.normal(0, 0.25, 3)
                            
                        # we ensure that lambda and sigma parameter respect bound condition (positive)
                        eta_init[5] = abs(eta_init[5]) # bound delta
                        eta_init[9] = abs(eta_init[9]) # bound sigma_Jr
                        eta_init[14] = abs(eta_init[14]) # bound sigma_Jsigma
                        eta_init[15] = abs(eta_init[15]) # bound sigma_JJsigma
                            
                        eta_init[16] = max(min(eta_init[16],1),-1) # bound rho_J
                        
                        eta_init[17] = abs(eta_init[17]) # bound lambda_r
                        eta_init[18] = abs(eta_init[18]) # bound lambda_sigma
                        eta_init[19] = abs(eta_init[19]) # bound lambda_r_sigma
                        
                        optimize_object = minimize(cost_function_param, eta_init, args=(W, sigma2_lst, lst_CM_matrix_estimator, used_moment), bounds=bounds, constraints=constraints, tol=10**(-4)) # SLSQP  solver by default. higher tolerence when no init
                        
                        for warning in captured_warnings:
                            if issubclass(warning.category, np.ComplexWarning):
                                print(f"Error with estimation retry : {warning.message}")
                        try_compute = False
                    
                except (RuntimeWarning,LinAlgError,Exception)  as e:  
                    print(f"Error with estimation retry : {e}")  
                    
            
            if optimize_object.fun < min_fun:
                print("improve opti fun by",round(min_fun-optimize_object.fun,5),"at step",r)
                eta_estim = optimize_object.x
                min_fun = optimize_object.fun
                
    else:
        min_fun = 10**10
        for r in range(0, sto_iter):  
            print("iter = ",r)
            eta_init_r = eta_init*(1+np.random.normal(0, 0.1, 20)) # perturbation of sd = 1%
            optimize_object = minimize(cost_function_param, eta_init_r, args=(W, sigma2_lst, lst_CM_matrix_estimator, used_moment), bounds=bounds, constraints=constraints, tol=10**(-6)) # SLSQP  solver by default
            if optimize_object.fun < min_fun:
                print("improve opti fun by",round(min_fun-optimize_object.fun,5),"at step",r)
                eta_init = optimize_object.x
                min_fun = optimize_object.fun
        eta_estim = eta_init
    
    return eta_estim




def calib_var_cov_param(eta, used_moment, nb_simul, nb_year, make_positive=True):
    # Param of cross moment estimator
    sigma2_lst = [0.1311, 0.2227,0.3041, 0.3913, 0.4922, 0.6169, 0.7829, 1.0328, 1.5018, 3.0984]
    h_list = [0.4165, 0.1811, 0.1374, 0.1173, 0.1126, 0.1131, 0.1239, 0.1496, 0.2134, 0.4397]
    
    # param for simulation
    mu_r = eta[0]
    rho_0 = eta[1]
    rho_1 = eta[2]
    m0 = eta[3]
    m1 = eta[4]
    delta = eta[5]
    mu_Jr = eta[6]
    mu_JJr0 = eta[7]
    mu_JJr1 = eta[8]
    sigma_Jr = eta[9]
    sigma_JJr0 = eta[10]
    sigma_JJr1 = eta[11]
    mu_Jsigma = eta[12]
    mu_JJsigma = eta[13]
    sigma_Jsigma = eta[14]
    sigma_JJsigma = eta[15]
    rho_J = eta[16]
    lambda_r = eta[17]
    lambda_sigma = eta[18]
    lambda_r_sigma = eta[19]
    
    lst_simul = []
    for simul in range(0,nb_simul):
        print("#################################")
        print("NB_SIMUL = ", simul+1, f"/ {nb_simul}")     
        print("#################################")
        T = nb_year * 252 # number of day of simulation
        nb_div_day = 390 # number minute in on day 
        P_init = 1000  # price init
        sigma_init = 1  # vol daily in percentage
        Pt, _ = simulation_full_model(T, nb_div_day, P_init, sigma_init, mu_r, rho_0, rho_1, m0, m1, delta, mu_Jr, mu_JJr0, mu_JJr1, sigma_Jr, sigma_JJr0, sigma_JJr1, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)


        lst_CM_matrix_estimator = []       
        for i in range(0, len(sigma2_lst)):
            estimated_matrix_cross_moment = estimation_matrix_estimator_cross_moment(np.sqrt(sigma2_lst[i]), h_list[i], Pt)
            corrected_cross_moment = corrected_matrix_estimator_cross_moment(estimated_matrix_cross_moment, 1) #correction in delta=1
            lst_CM_matrix_estimator.append(corrected_cross_moment)
            
            
        S = len(used_moment)
        G = len(sigma2_lst)       
        CM_estim = np.zeros((S*G,1))    
        for g in range(0,G):
            for s in range(0,S):
                p1 = used_moment[s,0]
                p2 = used_moment[s,1]
                CM_estim[g*S + s,0] = lst_CM_matrix_estimator[g][p1,p2]       
        lst_simul.append(CM_estim)
    
    simul_matrix = np.hstack(lst_simul)
    V = np.cov(simul_matrix, rowvar=True)
    
    if make_positive:   # we constrain our var cov matrix semi definite positive 
        eig_val, P = np.linalg.eig(V)
        index_to_correct = (eig_val.real <= 0) | (eig_val.imag != 0) # we identify all index were eig value are null, negative or complex
        eig_val[index_to_correct]=10**(-12)
        D = np.diag(eig_val)
        P_inv = np.linalg.inv(P)
        V_bis = P @ D @ P_inv
        V = (V_bis + V_bis.T)/2  # we make sure our matrix is symetric 

    return V, simul_matrix


def main_param_estimation(sto_iter_identity_param, sto_iter_opti_param, nb_simul_var_cov, nb_year_var_cov, lst_CM_matrix_estimator):
    
    used_moment = np.array([[1,0],[2,0],[3,0],[4,0],[0,1],[0,2],[0,3],[0,4],[1,1],[2,1],[3,1],[1,3],[1,2],[2,2],[3,2],[2,3]]) #cross moment used in the paper
    sigma2_lst = [0.1311, 0.2227,0.3041, 0.3913, 0.4922, 0.6169, 0.7829, 1.0328, 1.5018, 3.0984]
    
    
    S = len(used_moment)
    G = len(sigma2_lst)
    # Compute of param with identity to make simulation
    identity = np.eye(S*G)
    eta_opti_identity = parametric_estimation(identity, sigma2_lst, lst_CM_matrix_estimator, used_moment, sto_iter=sto_iter_identity_param)
    
    # Simulation of historical price data to compute empirical var-cov matrix
    V, matrix_simul_CM = calib_var_cov_param(eta_opti_identity,used_moment, nb_simul_var_cov, nb_year_var_cov)
     
    # Compte of optimal parameter with our var-cov matrix
    W = np.linalg.inv(V)
    eta_opti = parametric_estimation(W, sigma2_lst, lst_CM_matrix_estimator, used_moment, sto_iter = sto_iter_opti_param, eta_init=eta_opti_identity)

    return eta_opti, matrix_simul_CM


