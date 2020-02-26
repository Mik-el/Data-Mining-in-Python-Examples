#Per fare una statica completa dei posti letto per regione
#ci serve anche la popolazione per regione
#che non è presente nel database originario. 
#Quindi creiamo un nuovo database con: codice regione e popolazione. 
#Dobbiamo combinazione di due database.


import pandas as pd
import matplotlib.pyplot as plt

# Carica il primo csv 'postiletto1.csv'
dati = pd.read_csv( 'posti_letto.csv', sep=';' )
dati = dati.convert_objects( convert_numeric=True )

# selezione dei dati per anno 2011
anno = dati[ dati['Anno'] == 2011 ]

#???
letti = anno[ [ 'Codice Regione', 'Descrizione Regione', 'Totale posti letto' ] ]

#raggruppamento dei dati per codice regione
letti = letti.groupby( ['Codice Regione' ] )
#MIK
print ("\n **********\n Dati ragguppati per regione: \n *********\n")
print(letti)

#aggregazione dei dati (il numero di posti letto viene sommato)
letti = letti.aggregate( { 'Descrizione Regione':'first', 'Totale posti letto':'sum' } )
#MIK
print ("\n **********\n Dati aggregati: \n *********\n")
print(letti)





# Carica il secondo csv 'popolazione.csv'
dati2 = pd.read_csv( 'popolazione.csv', sep=';', thousands='.' )
dati2 = dati2.convert_objects( convert_numeric=True )
dati2n = dati2.rename( columns={'CODICE REGIONE': 'Codice Regione', 'TOTALE':'Popolazione'} )

# Raggruppo i cittadin per Codice regione e sommo tutti i loro cvalori. 
# Il risultato e'¨ la lista delle regioni con associato il numero di abitanti
popolazione = dati2[ ['Codice Regione', 'Popolazione' ] ].groupby( 'Codice Regione' ).sum()


## join tra i dataframe popolazione e letti. Serve almeno una variabile con lo stesso nome (nel nostro caso  Codice Regione).
lettiEPopolazione = popolazione.join( letti )

# Aggiungiamo una colonna: 'Letti per Cittadino' che e' data da 'Totale posti letto'/'Popolazione'
lettiEPopolazione['Letti per Cittadino'] = lettiEPopolazione['Totale posti letto'] / lettiEPopolazione['Popolazione']

# Ordinamento del nuovo dataframe 
ordinato = lettiEPopolazione.sort_values( 'Letti per Cittadino' )
print ("\n **********\n Dataframe ottenuto dal join e ordinato: \n *********\n")
print(ordinato)

print ("\n **********\n Statistiche descrittive dell' istanza Letti per Cittadino dopo il join: \n *********\n")
print (ordinato['Letti per Cittadino'].describe())




# Grafico, specifico tipo e cosa visuallizare sugli assi
ordinato.plot( kind='barh', x='Descrizione Regione', y='Letti per Cittadino' )
plt.show()


# la Lombardia, che sembrava possedere un numero spropositato di posti letto
#non è più neanche al primo posto. 
#Le differenze tra le varie regioni sono attenuate
#Il molise da ultimo per posti letto diventa il primo