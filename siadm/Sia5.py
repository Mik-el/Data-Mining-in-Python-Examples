import pandas as pd
import matplotlib.pyplot as plt
import csv

pd.options.mode.chained_assignment = None #

import time #time e datetime per operazioni sulle date
import datetime
import numpy as np


dataFrame = pd.read_excel("SIA_DM.xls")
pd.set_option('display.expand_frame_repr', False)

# RICHIESTA 5
# Verificare il campo CredCardUser, possibili errori di formattazione.
print('\n **********\n Verificare il campo CredCardUser, possibili errori di formattazione \n **********\n')

# Cattura l'eccezione durante la conversione ad intero per evidenziare errori di formattazione.
i = 0
count_err = 0
for tmp in dataFrame['CredCardUser']:
    try:
        int(tmp)
        i = i + 1
    except:
        print('Valore CredCardUser non riconosciuto', tmp)
        print('Nell\'istanza:', i+1)
        count_err = count_err + 1
        i = i + 1

if count_err == 0:
    print('Non ci sono errori di formattazione nell\'attributo CredCardUser')
    

#RICHIESTA 6    
# Verificare i campi Income e Purchases per missing values e outliers.


print('\n **********\n Elenco delle istanze con missing values per attributo Income: \n **********\n')
display(dataFrame[dataFrame['Income'].isnull()])

print('\n **********\n Elenco delle istanze missing values con attributo Purchases: \n **********\n')
display(dataFrame[dataFrame['Purchases'].isnull()])

# Per identificare i possibili outliers si considera la media e la deviazione standard
#print(dataFrame['Income'].mean())

    
    