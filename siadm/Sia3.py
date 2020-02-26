import pandas as pd
import matplotlib.pyplot as plt
import csv

pd.options.mode.chained_assignment = None #

import time #time e datetime per operazioni sulle date
import datetime
import numpy as np



dataFrame = pd.read_excel("SIA_DM.xls")
pd.set_option('display.expand_frame_repr', False)

# RICHIESTA 3
#Verificare se i valori del campo età (age) contengono valori improbabili.

print('Sto verificando e rimpiazzando gli errori nel campo età...')

# Cattura eccezioni sulla conversione ad intero per verificare la presenza di stringhe al posto di valori numerici
i = 0
for tmp in dataFrame['Age']:
    try:
        int(tmp)
        i = i + 1
    except:
        print('Età non riconosciuta', tmp)
        # Rimpiazza l'età errata con il valore 0
        dataFrame.loc[i, 'Age'] = 0
        i = i + 1

# Confronta l'età che dovrebbe avere nell'anno corrente rispetto all'età riportata nel data frame.
"""
i = 0
anno_corrente = datetime.date.today().year
for eta_letta in dataFrame['Age']:
    eta_corretta = anno_corrente - dataFrame.loc[i].Birthdate.year
    if eta_corretta != eta_letta:
        # Rimpiazza l'età errata con l'età corretta
        dataFrame.loc[i, 'Age'] = eta_corretta
    i = i + 1
"""
    
print('\n **********\n Stampa dataframe: \n **********\n')    
display(dataFrame)
    