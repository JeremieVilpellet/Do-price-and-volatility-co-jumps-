# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:36:54 2024

@author: 33768
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:10:33 2024

@author: 33768
"""
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd


mu_r = 0.036
rho0 = -0.0988
rho1 = -0.1617
m0= -0.0380
m1 = -0.0597
lambda_maj = 0.5583
mu_Jr = 1.3948
mu_JJr0= -0.0544
mu_JJr1=-1.0072
sigma_Jr=0.6818
sigma_JJr0=0.6246
sigma_JJr1=2.2469
mu_Jsigma=-0.4497
mu_JJsigma=1.4428
sigma_Jsigma=0.7002
sigma_JJsigma=0.1084 
rho_J=-1.00
lambda_r=0.0252
lambda_sigma= 0.0528
lambda_r_sigma= 0.0339


sigma_t = [0.36207734, 0.47191101, 0.55145263, 0.62553977, 0.70156967, 0.78542982, 0.88481637, 1.01626768, 1.2254795,  1.76022726]

# Valeur choisi dans l'annexe par sigma_t (sqrt de sigma_t2)

def rho_t(rho0, rho1, sigma_t):
    return np.max(np.minimum(rho0 + rho1*sigma_t, 1), -1)



def mu_JJr(mu_JJr0,mu_JJr1, sigma_t):
    return mu_JJr0 + mu_JJr1*sigma_t



def sigma_JJr(sigma_JJr0,sigma_JJr1, sigma_t):
    return (sigma_JJr0 + sigma_JJr1*sigma_t)**2


    
def return_RP(mu_r,r,sigma_t):
    # Fonction de calcul du return risk premium (Eq 17,5)
    return mu_r - r + 1/2*(sigma_t**2)



def mu_tilde(sigma_t,psy,phi):
    # Fonction de calcul de l'Eq 19 utilisee pour identifier avec l'Eq 17,5 phi et psy par regression non linéaire
    return (-phi*(sigma_t**2) -psy*(rho0 + rho1*sigma_t)*sigma_t*lambda_maj - lambda_r_sigma*np.exp(phi*(mu_JJr0+mu_JJr1*sigma_t)+ psy*mu_JJsigma + 0.5*phi**2*(sigma_JJr0+sigma_JJr1*(sigma_t))**2
                                                                                  +0.5*psy**2*sigma_JJsigma**2)*(np.exp(mu_JJr0 + mu_JJr1*sigma_t + 0.5*(sigma_JJr0 + 
                                                                                  + (sigma_JJr1*(sigma_t))**2)*(1+2*phi)+psy*rho_J*(sigma_JJr0+sigma_JJr1*(sigma_t))*sigma_JJsigma)
                                                                                  - 1) - lambda_r*np.exp(phi*mu_Jr + 0.5*(phi**2)*sigma_Jr**2)*(np.exp(mu_Jr + 0.5*(sigma_Jr**2)*(1+2*phi))-1))
                                                                                                                                                           
                                                                                                                                                           
def return_RP_composante_continu(sigma_t,psy,phi):
    return -phi*(sigma_t**2) -psy*(rho_t(rho0, rho1, sigma_t))*sigma_t*lambda_maj


def return_RP_co_saut(sigma_t,psy,phi):
    return lambda_r_sigma*(math.exp(phi*(mu_JJr0+mu_JJr1*sigma_t)+ psy*mu_JJsigma + 0.5*(phi**2)*(sigma_JJr0+sigma_JJr1*(sigma_t))**2 + 0.5*(psy**2)*(sigma_JJsigma**2))*(math.exp(mu_JJr0 + mu_JJr1*sigma_t + 0.5*((sigma_JJr0 + (sigma_JJr1*(sigma_t)))**2)*(1+2*phi)+psy*rho_J*(sigma_JJr0+sigma_JJr1*(sigma_t))*sigma_JJsigma)- 1))
                                                                                                                                                           
                                                                         
def return_RP_saut_idyosincratique(sigma_t,psy,phi):
    return lambda_r*math.exp(phi*mu_Jr + 0.5*(phi**2)*sigma_Jr**2)*(math.exp(mu_Jr + 0.5*(sigma_Jr**2)*(1+2*phi))-1)


                                                                   
def var_RP_Drift(sigma_t,psy,phi):
    # Fonction de calcul de la partie temporelle de variance risque premium
    # Developpement limité de Eq 33
    rhot = rho_t(rho0, rho1, sigma_t)
    return psy*lambda_maj**2 + phi*rhot*lambda_maj*sigma_t

def var_Rp_Saut_Co(sigma_t, psy, phi):
    # Fonction de calcul de la partie liee aux co-jumps de la variance risk premium
    # Calculee à partir de l'Eq 34
    return lambda_r_sigma*(psy*(sigma_JJsigma**2 + mu_JJsigma**2) + phi*(rho_J*sigma_JJsigma*(sigma_JJr0+sigma_JJr1*(sigma_t)) + mu_JJsigma*(mu_JJr0+mu_JJr1*sigma_t)))

def var_RP_Saut_Idio(sigma_t, psy, phi):
    # Fonction de calcul de la partie liee aux jumps idiosyncratique de la variance risk premium
    # Calculee à partir de l'Eq 35
    return lambda_sigma*psy*(sigma_Jsigma**2 + mu_Jsigma**2)


def delta_temporel(sigma_t, psy, phi):
    r = 0.03
    mu_chap = mu_tilde(sigma_t, psy, phi)
    return -r*(1+phi) - phi(mu_chap-0.5*sigma_t**2)-psy*(m0 + m1*math.log(sigma_t**2))-0.5*(phi**2*sigma_t**2 + psy**2*lambda_maj**2 + 2*phi*psy*rho_t*sigma_t*lambda_maj)-lambda_r_sigma*(math.exp(phi*(mu_JJr0+mu_JJr1*sigma_t)+psy*mu_JJsigma+0.5*(phi**2*(sigma_JJr0+sigma_JJr1*(sigma_t))**2+(psy**2)*sigma_JJsigma**2))-1) - lambda_r*(math.exp(phi*mu_Jr + 0.5*phi**2*sigma_Jr**2)-1) - lambda_sigma*(math.exp(psy*mu_Jsigma+0.5*psy**2*sigma_Jsigma**2)-1)

                                                                                                                                                  

def find_psi_and_phi(sigma_t):
    y = 0.03#r(sigma_t)
    params, covs = curve_fit(mu_tilde,sigma_t, y, bounds=([0.03, -2], [0.15, 2]))
    error = np.sqrt(np.diag(covs)) 
    return params, error



def graph_var_RP_Drift():
    psy = [0.057143 for i in range (10)]
    phi = [-1.7489 for i in range (10)]
    var_RP_Drift_ = [var_RP_Drift(sigma_t[i],psy[i],phi[i]) for i in range (10)]
    # Il faut multiplier par 100 pour avoir en %
    # Tracer le graphe
    plt.plot(sigma_t, var_RP_Drift_, marker='o', linestyle='-', color='blue')
    
    # Ajouter des titres et des labels d'axe
    plt.title('La composante continue de la variance risk premium en fonction de sigma_t en %')
    plt.xlabel('sigma (%, Daily)')
    plt.ylabel('Composante drift de la variance risk premium(%)')
    
    # Afficher le graphe
    plt.grid(True)
    plt.show()
    
    

def graph_var_Rp_Saut_Co():
    psy = [0.057143 for i in range (10)]
    phi = [-1.7489 for i in range (10)]
    var_RP_Drift_ = [var_Rp_Saut_Co(sigma_t[i],psy[i],phi[i]) for i in range (10)]
    # Il faut multiplier par 100 pour avoir en %
    # Tracer le graphe
    plt.plot(sigma_t, var_RP_Drift_, marker='o', linestyle='-', color='blue')
    
    # Ajouter des titres et des labels d'axe
    plt.title('role de la composante co saut dans la variance risk premium en fonction de sigma_t en %')
    plt.xlabel('sigma (%, Daily)')
    plt.ylabel('Composante co-saut de la variance risk premium')
    
    # Afficher le graphe
    plt.grid(True)
    plt.show()
    

def graph_var_Rp_Saut_Idyo():
    psy = [0.057143 for i in range (10)]
    phi = [-1.7489 for i in range (10)]
    var_RP_Drift_ = [var_RP_Saut_Idio(sigma_t[i],psy[i],phi[i]) for i in range (10)]
    # Il faut multiplier par 100 pour avoir en %
    # Tracer le graphe
    plt.plot(sigma_t, var_RP_Drift_, marker='o', linestyle='-', color='blue')
    
    # Ajouter des titres et des labels d'axe
    plt.title('role de la composante co saut dans la variance risk premium en fonction de sigma_t en %',  fontsize=10)
    plt.xlabel('sigma (%, Daily)')
    plt.ylabel('Composante idiosyncratique de la variance risk premium',  fontsize=7.8)
    
    # Afficher le graphe
    plt.grid(True)
    plt.show()



def graph_Return_RP_Drift():
    psy = [0.057143 for i in range (10)]
    phi = [-1.7489 for i in range (10)]
    var_RP_Drift_ = [return_RP_composante_continu(sigma_t[i],psy[i],phi[i]) for i in range (10)]
    # Il faut multiplier par 100 pour avoir en %
    # Tracer le graphe
    plt.plot(sigma_t, var_RP_Drift_, marker='o', linestyle='-', color='blue')
    
    # Ajouter des titres et des labels d'axe
    plt.title('La composante continue du return risk premium en fonction de sigma_t en %')
    plt.xlabel('sigma (%, Daily)')
    plt.ylabel('Composante continue du return risk premium(%)')
    
    # Afficher le graphe
    plt.grid(True)
    plt.show()


def graph_Return_RP_Co_Sauts():
    psy = [0.057143 for i in range (10)]
    phi = [-1.7489 for i in range (10)]
    var_RP_Drift_ = [return_RP_co_saut(sigma_t[i],psy[i],phi[i]) for i in range (10)]
    # Tracer le graphe
    plt.plot(sigma_t, var_RP_Drift_, marker='o', linestyle='-', color='blue')
    
    # Ajouter des titres et des labels d'axe
    plt.title('La composante des co-sauts du return risk premium en fonction de sigma_t en %')
    plt.xlabel('sigma (%, Daily)')
    plt.ylabel('Composante co-sauts du return risk premium(%)')
    
    # Afficher le graphe
    plt.grid(True)
    plt.show()



def graph_Return_RP_Idyosincratique():
    psy = [0.057143 for i in range (10)]
    phi = [-1.7489 for i in range (10)]
    var_RP_Drift_ = [return_RP_saut_idyosincratique(sigma_t[i],psy[i],phi[i]) for i in range (10)]
    # Tracer le graphe
    plt.plot(sigma_t, var_RP_Drift_, marker='o', linestyle='-', color='blue')
    
    # Ajouter des titres et des labels d'axe
    plt.title('La composante des sauts idyosyncratiques du return risk premium en fonction de sigma_t en %',fontsize=9 )
    plt.xlabel('sigma (%, Daily)')
    plt.ylabel('Composante sauts idyosyncratiques du return risk premium(%)', fontsize= 9)
    
    # Afficher le graphe
    plt.grid(True)
    plt.show()
    


# Fonction pour tracer le graphe combiné
def plot_combined_RVP_graph():
    psy = [0.057143 for i in range (10)]
    phi = [-1.7489 for i in range (10)]
    var_RP_Drift_vals = [var_RP_Drift(sigma_t[i],psy[i],phi[i]) for i in range (10)]
    var_Rp_Saut_Co_vals = [var_Rp_Saut_Co(sigma_t[i],psy[i],phi[i]) for i in range (10)]
    var_Rp_Saut_Idio_vals = [var_RP_Saut_Idio(sigma_t[i],psy[i],phi[i]) for i in range (10)]
    # Calcul de la Variance Risk Premium totale
    var_RP_total = [var_RP_Drift_vals[i] + var_Rp_Saut_Co_vals[i] + var_Rp_Saut_Idio_vals[i] for i in range(10)]

    # Tracé des graphes
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_t, var_RP_Drift_vals, marker='o', color='blue', label='Drift')
    plt.plot(sigma_t, var_Rp_Saut_Co_vals, marker='x', color='red', label='Co-saut')
    plt.plot(sigma_t, var_Rp_Saut_Idio_vals, marker='^', color='green', label='Idiosyncratique')
    plt.plot(sigma_t, var_RP_total, marker='s', color='purple', label='Variance RP Total')

    # Légendes et titres
    plt.title('Composantes de la Variance Risk Premium')
    plt.xlabel('sigma (%, Daily)')
    plt.ylabel('Valeur de la composante (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calcul et affichage des contributions
    total_contributions = [var_RP_total[i] for i in range(10)]
    contributions = {
        'Drift % ': sum([var_RP_Drift_vals[i] / total_contributions[i] if total_contributions[i] != 0 else 0 for i in range(10)]) * 10,
        'Co-saut % ': sum([var_Rp_Saut_Co_vals[i] / total_contributions[i] if total_contributions[i] != 0 else 0 for i in range(10)]) * 10,
        'Idiosyncratique % ': sum([var_Rp_Saut_Idio_vals[i] / total_contributions[i] if total_contributions[i] != 0 else 0 for i in range(10)]) * 10 
    }
    print(contributions)
    return contributions




def plot_combined_RP_graph():
    n = len(sigma_t)
    psy = [0.057143 for i in range (10)]
    phi = [-1.7489 for i in range (10)]
    RRP_Drift_vals = [return_RP_composante_continu(sigma_t[i],psy[i],phi[i]) for i in range (n)]
    RRP_Rp_Saut_Co_vals = [return_RP_co_saut(sigma_t[i],psy[i],phi[i]) for i in range (n)]
    RRP_Rp_Saut_Idio_vals = [return_RP_saut_idyosincratique(sigma_t[i],psy[i],phi[i]) for i in range (n)]
    # Calcul de la Variance Risk Premium totale
    PRP_total = [RRP_Drift_vals[i] + RRP_Rp_Saut_Co_vals[i] + RRP_Rp_Saut_Idio_vals[i] for i in range(n)]

    # Tracé des graphes
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_t, RRP_Drift_vals, marker='o', color='blue', label='Drift')
    plt.plot(sigma_t, RRP_Rp_Saut_Co_vals, marker='x', color='red', label='Co-saut')
    plt.plot(sigma_t, RRP_Rp_Saut_Idio_vals, marker='^', color='green', label='Idiosyncratique')
    plt.plot(sigma_t, PRP_total, marker='s', color='purple', label='Return RP Total')

    # Légendes et titres
    plt.title('Composantes du return Risk Premium')
    plt.xlabel('sigma (%, Daily)')
    plt.ylabel('Valeur de la composante (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calcul et affichage des contributions
    total_contributions = [PRP_total[i] for i in range(n)]
    contributions = {
        'Drift % ': sum([RRP_Drift_vals[i] / total_contributions[i] if total_contributions[i] != 0 else 0 for i in range(n)]) * 10,
        'Co-saut % ': sum([RRP_Rp_Saut_Co_vals[i] / total_contributions[i] if total_contributions[i] != 0 else 0 for i in range(n)]) * 10,
        'Idiosyncratique % ': sum([RRP_Rp_Saut_Idio_vals[i] / total_contributions[i] if total_contributions[i] != 0 else 0 for i in range(n)]) * 10 
    }
    print(contributions)
    return contributions



# Exécuter la fonction pour tracer le graphe et obtenir les contributions

plot_combined_RVP_graph()


params, error = find_psi_and_phi(sigma_t)


