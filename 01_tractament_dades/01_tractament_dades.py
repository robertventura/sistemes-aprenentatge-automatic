# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:39:38 2022

@author: robert
"""

#Importar dadades des d'un CSV
import pandas as pd

cotxes = pd.read_csv('dataset/venda-cotxes-segona-ma.csv', index_col=False)
cotxes.head()
