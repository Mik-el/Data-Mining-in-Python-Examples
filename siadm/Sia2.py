
import pandas as pd
import matplotlib.pyplot as plt
import csv

pd.options.mode.chained_assignment = None #

import time #time e datetime per operazioni sulle date
import datetime
import numpy as np


dataFrame = pd.read_excel("SIA_DM.xls")
pd.set_option('display.expand_frame_repr', False)

# RICHIESTA 2) verificare se le date di nascita contengono valori improbabili (possibili
#valori sconosciuti!!)
print('\n **********\n Verificare se le date di nascita contengono valori improbabili \n **********\n ')

# Come è possibile notare tra le informazioni del data frame, non sono presenti valori nulli.



#****DATE NON VALIDE
# Dunque si procede alla verifica di valori errati.
# In caso di errore nel campo data, questo verrà rimpiazzato da una data media
err_count = 0
idx = 0
err_idx_list = []     # lista utile a mantene in memoria gli indici corrispondenti ai valori errati
date_tmp_list = []    # lista utile a memorizzare le date per l'eventuale calcolo della media delle date
for tmp in dataFrame['Birthdate']:
    try:
        date_tmp_list.append(pd.to_datetime(tmp))
        idx = idx + 1
    except:
        print('Dato errato nell\'istanza numero', idx)
        err_idx_list.append(idx)
        err_count = err_count + 1
        idx = idx + 1

# Calcolo la data media per eventuali rimpiazzi
meanDate = (np.array(date_tmp_list, dtype='datetime64[s]').view('i8').mean().astype('datetime64[s]'))

# In caso di date errate si procede al rimpiazzo delle stesse con la data media
if err_count != 0:
    print('Esistono', err_count, 'date non corrette che verranno sostituite con la data media')
    for i in err_idx_list:
        # Stampa le istanze con le date non corrette
        display(dataFrame[dataFrame['Customer'] == i+1])
        dataFrame.loc[i,'Birthdate'] = meanDate
else:
    print('Tutte le date sono sintatticamente corrette')
    

# Una volta azzerati i dati errati, si procede con la conversione in "datetime64" per poter effettuare
# altre operazioni sulle date utili ai fini di altre verifiche
dataFrame['Birthdate'] = pd.to_datetime(dataFrame['Birthdate'])
display(dataFrame.head())





#****DATE IMPROBABILI
# ***clienti che abbiano la data di nascita a partire dal 1900.
# da 01-01-1900 ad oggi 
# Si crea una vista del data frame a tele scopo (df)
print('\n **********\n Si suppone di voler analizzare dati di clienti che abbiano la data di nascita a partire dal 1900.')
print('Dunque si assumono "probabili" solo date che ricadono nell\'intervallo temporale che va dal 01-01-1900 ad oggi. \n **********\n')
df = dataFrame[(dataFrame['Birthdate'] > time.strftime("%d/%m/%Y")) | (dataFrame['Birthdate'] < datetime.datetime(1900, 1, 1))]
total_err_date = df['Birthdate'].size

# Visualizza le istanze che non ricadono nell'intervallo di interesse
print('Le istanze che non ricadono nel range ricercato sono:', total_err_date)
display(df)
# Rimpiazzo le date improbabili con una data media.
# Essendo 'df' una vista sul data frame principale (dataFrame), i rimpiazzi effettuati hanno effetto su 'dataFrame'.
df.loc[((df['Birthdate'] > time.strftime("%d/%m/%Y")) | (df['Birthdate'] < datetime.datetime(1900, 1, 1))), 'Birthdate'] = meanDate
print('Le istanze con i valori improbabili sono stati rimpiazzati con una data media')
display(df)

print('Si nota ovviamente che il campo ""age" non combacia con la data di nascita')