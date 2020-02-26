# NUMERO DI LETTI PER REGIONE (IN UN ANNO)
#Raggruppo il conteggio dei letti per regione con il metodo groupBy, poi li sommo con sum.

import pandas as pd
import matplotlib.pyplot as plt
import csv

dati = pd.read_csv('posti_letto.csv', sep=';')
dati = dati.convert_objects(convert_numeric=True)

#dataset anno contenente solo voci relative all' anno 2014
anno2013 = dati[dati['Anno'] == 2013]

#del dataframe ci interessa solo la regione e il numero di posti letto 
lettiPerRegione = anno2013[  ['Descrizione Regione', 'Totale posti letto']  ].groupby('Descrizione Regione')


#dataframe ottenuto ordinando e sommando tutti i valori i postiletti relativi al dataset 
risultato = lettiPerRegione.sum().sort_values('Totale posti letto')


print ("\n **********\n Posti letto per regione nell' anno 2013: \n *********\n")
#invocando .plot.barh() sul dataset otteniamo un diagramma a barre orizziontali
risultato.plot.barh()

#invocando .show() su plt lo stampiamo
#plt.show()