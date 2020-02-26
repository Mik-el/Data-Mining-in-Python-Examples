import pandas as pd
import matplotlib.pyplot as plt
import csv

pd.options.mode.chained_assignment = None #

import time #time e datetime per operazioni sulle date
import datetime
import numpy as np


dataFrame = pd.read_excel("SIA_DM.xls")
pd.set_option('display.expand_frame_repr', False)

# RICHIESTA 4
#Verificare il campo region, possibili errori di digitazione.

print('Sto Verificando il campo region nel caso in cui ci fossero errori di digitazione.')

# Verifica se ogni istanza del data frame abbia un valore corretto nel campo 'Region'
# .lstrip() serve a ignorare gli spazi a sinistra della stringa.
i = 0
cont_err = 0
for reg in dataFrame['Region']:
    if (dataFrame.loc[i, 'Region'].lstrip() != 'North') & (dataFrame.loc[i, 'Region'].lstrip() != 'South') & (dataFrame.loc[i, 'Region'].lstrip() != 'East') & (dataFrame.loc[i, 'Region'].lstrip() != 'West'):
        print('Riga errata: ', i+1)
        cont_err = cont_err + 1
    i = i + 1
print('Vi sono', cont_err,'istranze errate nell\'attributo Region')


print('\n **********\n Stampa dataframe, dato che si tratta solo di una verifica non ho corretto i dati: \n **********\n')    
display(dataFrame)