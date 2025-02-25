# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:06:52 2024

@author: jvilp
"""

# =============================================================================
# Estimator or cross moment part 5 eq 12 with it's correction in delta and bias
# =============================================================================

import numpy as np
from math import pi, comb


def estimator_bipower_spot_variance(rdmt:list, knots=6):  # list of 65 * one minute log return so h=1/390
    range_lst = int(390/knots)
    if len(rdmt) != range_lst:
        raise ValueError(f"size liste rdmt must be {range_lst}")
    h=(1/390)  # freq of discretization not brandwith of kernel here
    threshold = h**0.99        # as in mancini 2009 this treshold seems to be more suitable than time varying threshold
    indic = [1 if abs(x) < threshold else 0 for x in rdmt]    
    nj = indic.count(0)
    zeta = 0.7979
    scale_factor = (range_lst/(range_lst-1-nj))*zeta**(-2)
    value_sum = 0
    
    for k in range(1,range_lst):
        value_sum = value_sum + abs(rdmt[k])*abs(rdmt[k-1])*indic[k]*indic[k-1]
        
    return scale_factor*value_sum

def estimator_unipower_spot_variance(rdmt:list, knots=6):
    range_lst = int(390/knots)
    if len(rdmt) != range_lst:
        raise ValueError(f"size liste rdmt must be {range_lst}")
    h=(1/390)  
    threshold = h**0.99        
    indic = [1 if abs(x) < threshold else 0 for x in rdmt]    
    nj = indic.count(0)
    zeta = 0.7979
    scale_factor = (range_lst/(range_lst-1-nj))*zeta**(-2)
    value_sum = 0
    
    for k in range(0,range_lst):
        value_sum = value_sum + (rdmt[k]**2)*indic[k]
        
    return scale_factor*value_sum
    


def gaussian_kernel(x):
    return (1/np.sqrt(2*pi))*np.exp(-0.5*x**2)


def EMA(value_t, ema_t_1, nb_lag):  # exponential moving average
    alpha = 2/(nb_lag+1)
    return alpha * value_t + (1-alpha)*ema_t_1



def get_K_V_P(sigma:float, h:float, lst_price:list):  # function to compute only one time Kernel, spot Variance and log Price
    N = 6   # nb_knots
    nb_minute_day = 390   #in one day
    nb_minute_knots = int(nb_minute_day/N)
    T = int(len(lst_price)/nb_minute_day)
    
    K = np.zeros((T,N))  # Kernel matrice
    P = np.zeros((T,N))  # Price matrice
    var_spot = np.zeros((T,N)) # Var spot estimator matrice
    smooth_var = np.zeros((T,N)) # Var smmothed for kernel
    
    for t in range(0, T):     # t varying from 0 to T-1
        for i in range(0,N):   # i varying from 0 to N-1
            P[t,i] = lst_price[t*nb_minute_day + i*nb_minute_knots + nb_minute_knots-1] 
            
            if t == 0 and i == 0:  # If we are at the end of price list we use last knots of the last day but minus one as we have n price for n rdmt          
                price_t_i_k = lst_price[t*nb_minute_day+i*nb_minute_knots : t*nb_minute_day+i*nb_minute_knots+nb_minute_knots+1]    #we select nb_minute_knot to compute nb_minute_knot rdmt
            else:
                price_t_i_k = lst_price[t*nb_minute_day+i*nb_minute_knots -1: t*nb_minute_day+i*nb_minute_knots+nb_minute_knots]
            rdmt_t_i = np.diff(np.log(price_t_i_k))
            var_spot[t,i] = estimator_bipower_spot_variance(rdmt_t_i) * 6 * 100**2 # As we estimate hourly variance we scale up to daily and put in percentage (vol*100 -> var * 100^2)
            
            if t == 0 :
                smooth_var[t,i] = var_spot[t,i]
            else:               
                smooth_var[t,i] = EMA(var_spot[t,i], smooth_var[t-1,i], 20)
                
            K[t,i] = gaussian_kernel((np.sqrt(smooth_var[t,i]) - sigma)/h)
            
    return K, var_spot, P
    

def estimator_cross_moment(p1:int, p2:int, K, V, P): # K, V and P are three matrix of a row for each day and a column for each knots containing kernel estimation, price and var spot estimation
    
    T, N = P.shape  # T = nb line and N nb of column
    numerator = 0
    denominator = 0
    c = 2.61/64   # variance of our spot variance estimator

    for t in range(0, T-1):     
        for i in range(0,N):             
            denominator = denominator + K[t,i]          
            
            diff_log_Pt = (np.log(P[t+1,i]) - np.log(P[t,i]))*100
            diff_log_Vt = (np.log(V[t+1,i]) - np.log(V[t,i]))
            
            if p2 == 0 or p2 == 1:
                numerator = numerator + K[t,i]*(diff_log_Pt**p1)*(diff_log_Vt**p2)
            if p2 == 2:
                numerator = numerator + K[t,i]*(diff_log_Pt**p1)*(diff_log_Vt**2 - 2*c)
            if p2 == 3:
                numerator = numerator + K[t,i]*(diff_log_Pt**p1)*(diff_log_Vt**3 - 6*c*diff_log_Vt)
            if p2 == 4:
                numerator = numerator + K[t,i]*(diff_log_Pt**p1)*(diff_log_Vt**4 - 12*c*diff_log_Vt**2 -12*c**2)
    
    return numerator/denominator



def estimation_matrix_estimator_cross_moment(sigma:float, h:float, lst_price:list):    
    CM_matrix = np.zeros((5, 5))
    K, V, P = get_K_V_P(sigma, h, lst_price)
    for p1 in range(0,5):
        for p2 in range(0,5):
            CM_matrix[p1,p2] = estimator_cross_moment(p1, p2, K, V, P)           
    return CM_matrix     


def corrected_estimator_cross_moment(p1:int, p2:int, delta:float, CM_matrix): 
    double_sum = 0  
    if p1 == 0 and p2 == 0:
        return 0
    if p1 == 1 and p2 == 0:
        return CM_matrix[1,0]
    if p1 == 0 and p2 == 1:
        return CM_matrix[0,1]
    
    for j1 in range(0,p1+1):
        for j2 in range(0,p2+1):
                if j1 == 0 and j2 == 0:
                    double_sum = 0
                elif j1 == p1 and j2 == p2:
                    double_sum = 0
                else:
                    double_sum = comb(p1,j1)*comb(p2,j2)*corrected_estimator_cross_moment(j1,j2, delta, CM_matrix)*corrected_estimator_cross_moment(p1-j1, p2-j2, delta, CM_matrix)
                
    return CM_matrix[p1,p2]-(delta/2)*double_sum
            
            
def corrected_matrix_estimator_cross_moment(CM_matrix, delta:float):
    CM_matrix_corrected = np.zeros((5, 5))
    for p1 in range(0,5):
        for p2 in range(0,5):
            CM_matrix_corrected[p1,p2] = corrected_estimator_cross_moment(p1, p2, delta, CM_matrix)           
    return CM_matrix_corrected   
            


                       

        
        
        
        
        
        
        
        
        
        
    

