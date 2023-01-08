# -*- coding: utf-8 -*-
#-- -----------------------------------------------------
#-- Implementació de l'algorisme Forward Selection
#-- per una regressió lineal múltiple
#--
#-- Autor: Robert Ventura
#-- -----------------------------------------------------

import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import pandas as pd

# Importem el dataset
df = pd.read_csv('dataset/50_Startups.csv')
df.head()

# Dividim el dataframe amb les variables independents (X) i les dependents (Y)
vars_x = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
var_y = 'Profit'

x = df[vars_x]
y = df[var_y]

# Construim les variables dummy a partir de la variable categòrica State
x = pd.get_dummies(x,columns=["State"],drop_first=True)
x.head()
vars_x.remove('State') # Traiem la vairable categòrica State

#Afegim una variable constant.
x = sm.add_constant(x)
#print(x)

vars_x = x.columns
vars_x = vars_x.to_list()

# Dividim el dataset amb dades de test i de train. 30% test 70% train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

## vars_x_opt és el conjunt de variables independents òptimes / significatives
## per predir la y.
vars_x_opt = []

SL = 0.05
fi = False
print(vars_x)
while(len(vars_x)==0 or not fi):
    x_pvalors = dict()
    
    # Calculem tots els models amb les variables x que ens quedin.
    # Per construir un diccionari a on hi hagi per cada VI el pvalor.
    for vi_indep in vars_x :
        vars_x_tmp = vars_x_opt.copy() # Variable temporal per no utilitzar vars_x_opt
        vars_x_tmp.append(vi_indep)
        # Generem el model     
        print("\nGenerem el model per:")
        print(vars_x_tmp)
        model = sm.OLS(endog = y_train, exog = x_train[vars_x_tmp])
        results = model.fit()
        pvalors = results.pvalues.to_dict()
        
        x_pvalors[vi_indep] = pvalors[vi_indep]
        print("La variable '{}' té un p-valor {}".format(vi_indep,pvalors[vi_indep]))        
    
    # Un cop calculats tots els models ens quedarem amb el model 
    # que té el p-valor mínim.    
    min_pvalor =  min(x_pvalors.values())
    min_pvalor_key = min(x_pvalors, key=x_pvalors.get)
    
    if min_pvalor < SL:
        #Afegim la VI a la llista de var_x_opt
        vars_x_opt.append(min_pvalor_key)        
        #Traiem de la llista de candidats
        print("-------")
        print("------> Escollim la variable '{}' amb un p-valor {}".format(min_pvalor_key,min_pvalor))
        print("-------")
        #print(vars_x[0])    
        vars_x.remove(min_pvalor_key)
    else:
        #Finalitzem perquè ja no hi ha cap model amb pvalor < SL
        fi = True

#Un cop hem finalitzat tenim les variables VI òptimes. Per tant calculem el model amb aquestes
model = sm.OLS(endog = y_train, exog = x_train[vars_x_opt])
results = model.fit()
pvalors = results.pvalues.to_dict()

print(results.summary())