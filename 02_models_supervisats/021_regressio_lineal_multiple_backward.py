#-- -----------------------------------------------------
#-- Implementació de l'algorisme Backward Elimination
#-- per una regressió lineal múltiple
#--
#-- Autor: Robert Ventura
#-- -----------------------------------------------------

#import statsmodels.formula.api as sm
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

# Dividim el dataset amb dades de test i de train. 30% test 70% train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

## x_opt és el conjunt de variables independents òptimes / significatives
## per predir la y.
x_train_opt =  x_train.iloc[:,:]


SL = 0.05

while(True):
    # Generem el model     
    model = sm.OLS(endog = y_train, exog = x_train_opt)
    results = model.fit()
    pvalors = results.pvalues.to_dict()
    
    max_pvalor =  max(pvalors.values())
    max_pvalor_key = max(pvalors, key=pvalors.get)
    if max_pvalor > SL:
        #Traiem la variable independent que que no es prou significativa
        x_train_opt = x_train_opt.drop([max_pvalor_key], axis=1)
    else:
        break

print(results.summary())