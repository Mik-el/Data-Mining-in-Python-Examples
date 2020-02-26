#RELAZIONE TRA NUMERO DI POSTI LETTO E NUMERO DI OSPEDALI

import pandas as pd
import matplotlib.pyplot as plt
import csv

dati = pd.read_csv('posti_letto.csv', sep=';')
dati = dati.convert_objects(convert_numeric=True)

#CHECK dei valori 2011
anno2011 = [dati['Anno'] == 2011]
#senza [] anno2011 è un oggetto di tipo series
print ("\n **********\n Stampa dataset originale con ospendali censiti nel 2011 \n *********\n")
display(anno2011)
print ("\n **********\n Il datatype di anno2011 è: \n *********\n")
print(type(anno2011))



#ASSEGNAZIONE
#con [] anno2011 viene castato a string
anno2011_2 = dati[dati['Anno'] == 2011]

print ("\n **********\n Stampa ospedali censiti nel 2011 Inseriti in un nuovo dataset \n *********\n")
display(anno2011_2)


#dataframe con una sola colonna
letti = anno2011_2['Totale posti letto']
print ("\n **********\n Il datatype di letti è: \n *********\n")
print(type(anno2011))

print ("\n **********\n Stampa statistiche descrittive sul dataset letti \n *********\n")
print(letti.describe())

print ("\n **********\n Stampa istogramma con 30 colonne \n *********\n")
istogramma = letti.hist(bins=30)

istogramma.set_title('Relazione tra numero di letti e numero di ospedali')
istogramma.set_xlabel('Numero letti')
istogramma.set_ylabel('Numero ospedali')

plt.show()

#tanti ospedali hanno pochi posti letto
#e pochi ospedali con tanti posti letto
#senza il grafico la media non è "affidabile"
# per questo il dato della media va sempre rapportato ad una rappresentazione grafica
#da sola la media è un dato poco rappresentativo della realtà.

