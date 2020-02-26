"""" commento multilinea"""

""""
ESPLORARE UN DATASET di posti letto
"""
#pd.set_option('display.float_format', True)
#pd.set_option('display.max_columns', None)  

#pd.set_option('display.expand_frame_repr', False)
#pd.set_option('max_colwidth', -1)

#pd.options.display.width = None

#pd.options.display.max_rows = 1000


import pandas as pd
import matplotlib.pyplot as plt
import csv

#pd.set_option('display.expand_frame_repr', False)

pd.options.display.max_columns = None
pd.options.display.width=None

dati = pd.read_csv('posti_letto.csv', sep=';')

print(dati.info())
dati = dati.convert_objects(convert_numeric=True)

print(dati.info())
print ("\n **********\n Stampa le prime 5 righe del file \n *********\n")
#with pd.option_context('expand_frame_repr', False):
#    print(dati.head(5))


print ("\n **********\n Stampa il data type per ogni colonna \n *********\n")
#print(dati.dtypes)

print ("\n **********\n Stampa statistiche descrittive \n *********\n")
#print(dati.describe)



#documentazione pandas https://pandas.pydata.org/pandas-docs/stable/










