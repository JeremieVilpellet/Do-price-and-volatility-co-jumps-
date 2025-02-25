# error handling so as not to stop the code with large simulation
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
from numpy.linalg import LinAlgError
from contextlib import contextmanager

@contextmanager
def catch_complex_warning():
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", np.ComplexWarning)  
        yield captured_warnings

# import library
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from cross_moment_estimation import cross_moment, implied_matrix_cross_moment
from cross_moment_estimator import get_K_V_P




def cost_function_non_param(g, W, sigma, CM_matrix_estimator, used_moment): # g are component of sde system
        
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
    
    S = len(used_moment)
      
    CM_true = np.zeros((S,1))
    CM_estim = np.zeros((S,1))
    
    for i in range(0,S):
        p1 = used_moment[i,0]
        p2 = used_moment[i,1]
        CM_true[i,0] = cross_moment(p1, p2, sigma, mu_r, rho, m, delta, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)
        CM_estim[i,0] = CM_matrix_estimator[p1,p2]
    
    bias = CM_estim - CM_true
    cost_function = np.dot(bias.T,np.dot(W,bias))
    
    return cost_function[0,0]
        
    
    
def non_parametric_estimation(W, sigma, CM_matrix_estimator, used_moment, g_init = None, sto_iter = None):  # for first estimation of parameter with identity try a high sto_iter
    bounds = [(None, None), # mu_r 
              (-1, 1),      # rho
              (None, None), # m
              (0, None),    # delta
              (None, None), # mu_Jr
              (None, None), # mu_JJr
              (0, None),    # sigma_Jr
              (0, None),    # sigma_JJr
              (None, None), # mu_Jsigma
              (None, None), # mu_JJsigma
              (0, None),    # sigma_Jsigma
              (0, None),    # sigma_JJsigma
              (-1, 1),      # rho_J
              (0, None),    # lambda_r
              (0, None),    # lamda_sigma
              (0, None)]    # lambda_r_sigma


    if g_init is None:
        min_fun = 10**10
        for r in range(0, sto_iter):  
            
            g_init = np.random.normal(0, 0.25, 16) # random init param normal law N(0,0.25) (fixed len=16 as fix number of component in our non parametric sde system)
            
            # we ensure that our stochastic parameter respect bound condition
            g_init[1] = max(min(g_init[1],1),-1)   # bound rho 
            g_init[12] = max(min(g_init[12],1),-1) # bound rho_J
            
            g_init[3] = abs(g_init[3]) # bound delta
            g_init[6] = abs(g_init[6]) # bound sigma_Jr
            g_init[7] = abs(g_init[7]) # bound sigma_JJr
            g_init[10] = abs(g_init[10]) # bound sigma_Jsigma
            g_init[11] = abs(g_init[11]) # bound sigma_JJsigma
            
            g_init[13] = abs(g_init[13]) # bound lambda_r
            g_init[14] = abs(g_init[14]) # bound lambda_sigma
            g_init[15] = abs(g_init[15]) # bound lambda_r_sigma
            
            try:
                optimize_object = minimize(cost_function_non_param, g_init, args=(W, sigma, CM_matrix_estimator, used_moment), bounds=bounds, tol=10**(-4)) # L-BFGS-B solver by default. higher tolerence when no init
            except RuntimeWarning as e:
                print(f"RuntimeWarning with random init param at iteration {r}: {e}")
                continue
            except Exception  as e: 
                print(f"Error with random init param at iteration {r}: {e}")
                continue 
            
            if optimize_object.fun < min_fun:
                print("improve opti fun by",round(min_fun-optimize_object.fun,5),"at step",r)
                g_estim = optimize_object.x
                min_fun = optimize_object.fun
                
    else:       
        optimize_object = minimize(cost_function_non_param, g_init, args=(W, sigma, CM_matrix_estimator, used_moment), bounds=bounds, tol=10**(-6)) # L-BFGS-B solver by default
        g_estim = optimize_object.x
        
    return g_estim
    


def compute_L(sigma:float, h:float, lst_price:list):
    K, _, _ = get_K_V_P(sigma, h ,lst_price)
    L = K.sum()/h
    return L



def calib_var_cov_non_param(g, sigma, h, L, used_moment, make_positive=True): # g is the set of functions found with minimisation with identity matrix for a given sigma level,  h and L are bandwith and estimate of vol density for a sigma level
    
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
    
    S = len(used_moment)
    V = np.zeros((S,S))
    
    for i in range(0,S):
        for j in range(0,S):
            p1, p2 = used_moment[i]
            p3, p4 = used_moment[j]
            f_i_j = cross_moment(p1+p3, p2+p4, sigma, mu_r, rho, m, delta, mu_Jr, mu_JJr, sigma_Jr, sigma_JJr, mu_Jsigma, mu_JJsigma, sigma_Jsigma, sigma_JJsigma, rho_J, lambda_r, lambda_sigma, lambda_r_sigma)
            V[i,j] = f_i_j/(h*L)
    
    if make_positive:   # we constrain our var cov matrix semi definite positive 
        eig_val, P = np.linalg.eig(V)
        eig_val[eig_val<=0]=10**(-12)
        D = np.diag(eig_val)
        P_inv = np.linalg.inv(P)
        V_bis = P @ D @ P_inv
        V = (V_bis + V_bis.T)/2  # we make sure our matrix is symetric 

    
    return V
    

def main_non_param_estimation(Pt, lst_CM_matrix_estimator, sto_iter, nb_iter_estimation):  # Pt : list of price minute by minute, lst_CM_matrix_estimator : list of cross moment matrix for various vol level, sto_iter : number of random init param to try, nb_iter_estimation : number of iteration of estimation procedure to compute confidence interval  
    
    used_moment = np.array([[1,0],[2,0],[3,0],[4,0],[0,1],[0,2],[0,3],[0,4],[1,1],[2,1],[3,1],[1,3],[1,2],[2,2],[3,2],[2,3]])    #cross moment used in the paper
    names_component=["mu_r", "rho", "m", "delta", "mu_Jr", "mu_JJr", "sigma_Jr", "sigma_JJr", "mu_Jsigma", "mu_JJsigma", "sigma_Jsigma", "sigma_JJsigma", "rho_J", "lambda_r", "lambda_sigma", "lambda_r_sigma"]
    sigma2_lst = [0.1311, 0.2227,0.3041, 0.3913, 0.4922, 0.6169, 0.7829, 1.0328, 1.5018, 3.0984]
    h_list = [0.4165, 0.1811, 0.1374, 0.1173, 0.1126, 0.1131, 0.1239, 0.1496, 0.2134, 0.4397]
    
    lst_estimation = []  
      
    for nb_iter in range(0, nb_iter_estimation):
        
        print("#################################")
        print("NB_ITER = ", nb_iter+1, f"/ {nb_iter_estimation}")     
        print("#################################")
        df_non_param_component = pd.DataFrame()
        
        for i in range(0, len(sigma2_lst)):
            sigma=np.sqrt(sigma2_lst[i])
            h = h_list[i]
            CM_matrix_estimator = lst_CM_matrix_estimator[i]
            identity = np.eye(len(used_moment))   
            print("non param estimation for vol=",round(sigma,4))
            
            # computation of optimal component accounting of var-cov matrix of estimator
            try_compute = True
            while try_compute:  # we catch error in case of our estimation of g component with identity was not good
                try:
                    with catch_complex_warning() as captured_warnings:
                        
                        # computation of component value with linear system for a sigma level to compute var cov matrix of cross momen
                        g_opti_identity = non_parametric_estimation(identity, sigma, CM_matrix_estimator, used_moment, sto_iter = sto_iter)
                        
                        # calibration var-cov matrix with linear component
                        L = compute_L(sigma,h,Pt)
                        V = calib_var_cov_non_param(g_opti_identity, sigma, h, L, used_moment)
                        W = np.linalg.inv(V)
                        
                        # computation of component taking account var-cov 
                        g_opti = non_parametric_estimation(W, sigma, CM_matrix_estimator, used_moment, g_init = g_opti_identity) 
                                         
                        
                        for warning in captured_warnings:
                            if issubclass(warning.category, np.ComplexWarning):
                                print(f"Error with estimation retry : {warning.message}")
                        try_compute = False
                    
                except (RuntimeWarning,LinAlgError,Exception)  as e:  
                    print(f"Error with estimation retry : {e}")                  
            
            # load in df and add to list of estimation
            df_non_param_component[f"vol={round(sigma,3)}"]=pd.Series(g_opti,index=names_component)
            
        lst_estimation.append(df_non_param_component)
                     
    # compute the mean of all estimation to have consistent estimation
    mean_estimation_non_param_component = pd.concat(lst_estimation, axis=0).groupby(level=0).mean()
    
    # compute of all implied cross moment with mean of optimal parameter
    lst_implied_cross_moment = []
    for i in range(0, len(sigma2_lst)):
        
        sigma=np.sqrt(sigma2_lst[i])
        g_opti = mean_estimation_non_param_component[f"vol={round(sigma,3)}"].to_numpy()
        
        implied_CM = implied_matrix_cross_moment(g_opti, sigma)
        lst_implied_cross_moment.append(implied_CM)

    return mean_estimation_non_param_component, lst_estimation, lst_implied_cross_moment

