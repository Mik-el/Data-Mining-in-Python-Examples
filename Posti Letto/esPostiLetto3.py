#OSPEDALE CON PIU' POSTI LETTO
import pandas as pd
import matplotlib.pyplot as plt
import csv



dati = pd.read_csv('posti_letto.csv', sep=';', encoding= "UTF-8")

anno2001 = dati[dati['Anno'] == 2011]
#anno = dati['Anno'] == 2011
#anno = dati #OK
anno2001 = anno2001.sort_values('Totale posti letto', ascending=False)

pd.option_context('display.max_rows', None, 'display.max_columns', None)


#stampa i soli attributi 'Denominzione struttura' e 'Totale posti letto' del dataset anno
print(anno[['Denominazione struttura', 'Totale posti letto']])

