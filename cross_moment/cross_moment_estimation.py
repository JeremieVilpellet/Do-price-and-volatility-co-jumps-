# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:26:08 2024

@author: jvilp
"""

# =============================================================================
# For non parametric and parametric estimation of cross moment 5.2 
# Estimation of var-cov matrix 
# =============================================================================


from math import comb
import numpy as np



def double_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * double_factorial(n - 2)


def G(g1,g2,rho_J):
    if g1 < 0 or g2 < 0:
        return 0
    if g1 == 0 and g2 == 0:
        return 1
    
    elif g1 == 0:
        if g2%2 == 1:
            return 0
        else:
            return double_factorial(g2/2 -1)
    elif g2 == 0:
        if g1%2 == 1:
            return 0
        else:
            return double_factorial(g1/2 -1)
    else:
        return (g1+g2-1)*rho_J*G(g1-1,g2-1,rho_J) + (g1-1)*(g2-1)*G(g1-2,g2-2,rho_J)
    
        
def cross_moment(p1, p2, sigma, mu_r, rho, m, delta, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma):
    CM = 0
    if p1 == 0:
        if p2 == 1:
            CM = m + cross_moment_jump(0, p2, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)
        elif p2 == 2:
            CM = delta**2 + cross_moment_jump(0, p2, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)
        else:
            CM = cross_moment_jump(0, p2, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)
    
    elif p2 == 0:
        if p1 == 1:
            CM = mu_r + cross_moment_jump(p1, 0, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)
        elif p1 == 2:
            CM = sigma**2 + cross_moment_jump(p1, 0, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)
        else:
            CM = cross_moment_jump(p1, 0, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)
    
    elif p1 == 1 and p2 == 1:
        CM = rho*delta*sigma + cross_moment_jump(p1, p2, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)

    else:
        CM = cross_moment_jump(p1, p2, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)

    return CM



def cross_moment_jump(p1, p2, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma):
    CM_jump = 0
    if p1 == 0:
        for j in range(0,p2+1):
            CM_jump_j_1 = lambda_r_sigma*(comb(p2,j)*G(0,j,rho_J)*(sigma_JJsigma**j)*(mu_JJsigma**(p2-j))) # two part of the sum
            CM_jump_j_2 = lambda_sigma*(comb(p2,j)*G(0,j,rho_J)*(sigma_Jsigma**j)*(mu_Jsigma**(p2-j)))
            
            CM_jump = CM_jump + CM_jump_j_1 + CM_jump_j_2
            
    elif p2 == 0:      # same as for p1 = 0 but we use parameter of price returns jumps
        for j in range(0, p1+1):
            CM_jump_j_1 = lambda_r_sigma*(comb(p1,j)*G(0,j,rho_J)*(sigma_JJr**j)*(mu_JJr**(p1-j))) # two part of the sum
            CM_jump_j_2 = lambda_r*(comb(p1,j)*G(0,j,rho_J)*(sigma_Jr**j)*(mu_Jr**(p1-j)))
            
            CM_jump = CM_jump + CM_jump_j_1 + CM_jump_j_2
    else:
        for j1 in range(0,p1+1):
            for j2 in range(0,p2+1):
                CM_jump = CM_jump + lambda_r_sigma*comb(p1,j1)*comb(p2,j2)*G(j1,j2,rho_J)*(sigma_JJr**j1)*(sigma_JJsigma**j2)*(mu_JJr**(p1-j1))*(mu_JJsigma**(p2-j2))
                       
    return CM_jump



def implied_matrix_cross_moment(g, sigma):
    mu_r = g[0]
    rho = g[1]
    m = g[2]
    delta = g[3]
    mu_Jr = g[4]
    mu_JJr = g[5]
    sigma_Jr = g[6]
    sigma_JJr = g[7]
    mu_Jsigma = g[8]
    mu_JJsigma = g[9]
    sigma_Jsigma = g[10]
    sigma_JJsigma = g[11]
    rho_J = g[12]
    lambda_r = g[13]
    lambda_sigma = g[14]
    lambda_r_sigma = g[15]
    
    CM_matrix = np.zeros((5, 5))
    for p1 in range(0,5):
        for p2 in range(0,5):
            CM_matrix[p1,p2] = cross_moment(p1, p2, sigma, mu_r, rho, m, delta, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)          
    return CM_matrix 



    
    





