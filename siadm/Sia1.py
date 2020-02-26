
#Il file SIA_DM.XLS contiene i dati di 1500 clienti di un azienda

import pandas as pd
import matplotlib.pyplot as plt
import csv

pd.options.mode.chained_assignment = None #

import time #time e datetime per operazioni sulle date
import datetime
import numpy as np






dataFrame = pd.read_excel("SIA_DM.xls")     
pd.set_option('display.expand_frame_repr', False)

print('\n **********\n Visualizzazione delle prime 5 righe del data frame \n **********\n')
display(dataFrame.head())
print('\n')

print('\n **********\n Statistiche descrittive \n **********\n')
display(dataFrame.info()) #.describe()

print('\n **********\n Il numero di righe e di colonne è:\n **********\n', dataFrame.shape)

#Stampa del tipo di attributi
print ("\n **********\n Stampa del datatype di ogni colonna: \n *********\n") 
display(dataFrame.dtypes)



# RICHIESTA 1) verificare che i Social Security Numbers siano tutti diversi
# Verificare che Social Security Numbers siano tutti diversi

print('\n **********\n Verifica che Social Security Numbers siano tutti diversi \n **********\n')

ssn_totali = dataFrame['SSN'].size
ssn_unici = dataFrame['SSN'].unique().size
ssn_differenza = ssn_totali - ssn_unici
if ssn_differenza != 0:
    print('Sono presenti', ssn_differenza,'SSN che si ripetono, ovvero i seguenti:')
    display(dataFrame['SSN'].value_counts().iloc[0:ssn_differenza])
else:
    print('Tutti i Social Security Numbers sono diversi')












#3)verificare se i valori del campo eta` (age) contengono valori
#improbabili (problema “anno 2000”)



#4)verificare il campo region (possibili errori di digitazione!!)



#5)verificare il campo CredCardUser (possibile errore di formattazione)




#6)Verificare i campi Income e Purchases per missing values e outliers
