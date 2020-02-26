# Il dataset Bank Data contiene informazioni bancarie e personali di alcuni clienti.
#Si vogliono definire modelli di  targeting per classificare i clienti e vedere la distribuzine del reddito, in base all'acquisto del PEP (Personal 
# Equity Plan)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Caricamento del dataset 
data = pd.read_csv("bank-data.csv", sep=';')
pd.set_option('display.expand_frame_repr', False)

print ("\n **********\n Stampa del n. tot di dati del n. tot di attributi: \n *********\n")
print(data.shape)
print ("\n **********\n Stampa delle prime 5 righe del dataset: \n *********\n")
display(data.head()) 
print ("\n **********\n Stampa delle statistiche descrittive: \n *********\n")
display(data.describe())
print ("\n **********\n Stampa del datatype di ogni colonna: \n *********\n") 
display(data.dtypes)




# PREPROCESSING
# Normalizzazione delle fasce di età: 
# ● 0 = intervallo [18, 40] 
# ● 1 = intervallo [40, 60] 
# ● 2 = intervallo [60, 80] 
# ● 3 = intervallo [80, 100]

#prendo le istanze di tipo 'age' dal dataset data
age = data['age']
#metto quei valori in una lista
title = pd.Series(["age"])
#con la funzione .cut vado a specificare i range per l' attributo
new_age = pd.cut(age, [18, 40, 60, 80, 100], labels=False, retbins=False, right=False)
new_age = title.append(new_age)
print ("\n **********\n Stampa delle nuove fasce d'età per ogni persona: \n *********\n") 
print(new_age)

print ("\n **********\n Istogramma delle nuove fasce d'età per ogni persona: \n *********\n")

 

#apro il dataset age.csv, inserisco la nuova colonna new_age e lo chiudo
f = open("age.csv", "a") #'a' crea un nuovo file se non esiste
new_age.to_csv('bank-data.csv', index = False )
f.close()

data_con_newage = pd.read_csv("bank-data.csv", sep=';')
display(data_con_newage.head())

"""
ist1 = data_con_newage['new_age'].hist(bins=5, flagsize=(10,5))
plt.show()#
#.plot.barh()
"""


#non ci sono dati mancanti.
#altrimenti usavo ad esempio data = data.fillna(0)

#dati eterogeneri e tanti dati categoriali. 
#DISCRETIZZAZIONE: uniformare i dati categoriali trasformandoli in numerici
 #conversione in valori interi
data1 = data.copy(deep = True)
to_be_encoded_cols = data1.columns.values


def label_encode(df, columns):
    for col in columns:
        le = LabelEncoder()
        col_values_unique = list(df[col].unique())
        le_fitted = le.fit(col_values_unique)
        
        col_values = list(df[col].values)
        le.classes_
        col_values_transformed = le.transform(col_values)
        df[col] = col_values_transformed
        
to_be_encoded_cols = data.columns.values
label_encode(data, to_be_encoded_cols)


print ("\n **********\n Stampa delle prime 5 righe del dataset dopo il preprocessing: \n *********\n")
display(data.head(5)) 
print ("\n **********\n Stampa delle statistiche descrittive del dataset dopo il preprocessing: \n *********\n")
display(data.describe())


#ISTOGRAMMA
"""print ("\n **********\n Stampa istogramma con 30 colonne \n *********\n")
istogramma = data.hist(bins=30)

istogramma.set_title('Relazione tra guadagno dei clienti e loro regione di provenienza')
istogramma.set_xlabel('egion')
istogramma.set_ylabel('income')

plt.show()
"""
#ISTOGRAMMA 2
#print ("\n **********\n Stampa istogramma \n *********\n")
#data.plot.barh()



#apriamo il file bank-data2, ci andiamo a copiare il dataset discretizzato
f = open("bank-data2.csv", "a")
data.to_csv('bank-data2.csv', index = False )
f.close()

# a questo punto rimuovo manualmente l’intera colonna dell’attributo ‘id’, in quanto non è utile ai ﬁni della nostra analisi
#e sostituiamo la colonna 'age' con quella appena creata.
#PREPROCESSING FINITO

# ANALISI GRAFICA dei dati
# regione abitativa INNER_CITY (centro città), TOWN (città), RURAL (campagna), SUBURBAN (periferia).
# Mi è utile pcomparare la statistiche (describe) sul reddito dei clienti in base alla regione.
# poi costruisco un istogramma per vedere come la variabile ‘income’ viene distribuita.
region = data[['region', 'income']].groupby('region')
somma = region.sum().sort_values('income')
somma.plot.barh()
print ("\n **********\n Statistiche sul reddito dei clienti in base alla regione abitativa: \n *********\n")
print(somma.describe())
print ("\n **********\n E relativo istogramma: \n *********\n")
plt.show()

# Per fare l’analisi graﬁca ho dovuto effettuare una discretizzazione preliminare
# metto una legenda per interpretare i dati dell’istogramma. 
print (" ● 0 = inner_city (centro città) \n ● 1 = rural (campagna) \n ● 2 = suburban (periferia) \n ● 3 = town (città)")


# Vedo che il reddito oscilla da 20mila a 80mila.
# Su 600 clienti circa la metà ha un reddito di circa 42000.00.
# Dall’istogramma, invece, noto che i clienti con 
# reddito più alto sono concentrati nel centro città, mentre quelli con reddito più basso in periferia.

 
# ANALISI SIMILI possono essere fatte per  valutare ad esempio 
# distribuzione del reddito in base al matrimonio e al numero dei ﬁgli.
# o altri attributi a disposizione nel dataset

children = data[ data['children'] == 0 ]
som = children[ [ 'region', 'income', 'married' ] ]
som = som.groupby( ['region' ] )
som = som.aggregate( { 'income':'first', 'married': 'sum' } )
som.plot( kind='barh', x='income', y='married' )

print ("\n **********\n Statistiche sul reddito dei clienti senza figli: \n *********\n")
print(som.describe())
print ("\n **********\n E relativo istogramma: \n *********\n")
plt.show()


# Distribuzione reddito di clienti sposati e con due ﬁgli.
children = data[ data['children'] == 2 ]
som = children[ [ 'region', 'income', 'married' ] ]
som = som.groupby( ['region' ] )
som = som.aggregate( { 'income':'first', 'married': 'sum' } )
som.plot( kind='barh', x='income', y='married' )
print(som.describe())
plt.show()

print ("\n **********\n Statistiche sul reddito dei clienti sposati e con 2 figli: \n *********\n")
print(som.describe())
print ("\n **********\n E relativo istogramma: \n *********\n")
plt.show()




# ANALISI DEI DATI
# Matrice correlaz con Variabile Target la variabile Pep
correlation_matrix = data.corr()
plt.figure(figsize=(50,20))
ax = sns.heatmap(correlation_matrix, vmax=1, square=True, 
                 annot=True, fmt='.2f', cmap='GnBu', cbar_kws={"shrink":.5}, robust=True)
plt.title('Matrice di correlazione delle features', fontsize=20)
print ("\n **********\n Matrice di correlazione: \n *********\n")
plt.show()

# La matrice di correlazione mostra che l'unica variabile correlata con la target
# è 'income'(reddito del cliente), anche se la correlazione è comunque bassa (0.21)
# Si può notare però che le due variabili in maggior correlazione sono income ed age (0.17).


# Per un dataset con  molte funzionalità potrebbe diventare molto grande e la 
# correlazione di una singola funzione con le altre caratteristiche diventa diﬃcile da discernere.
#Se si desidera esaminare le correlazioni di una singola funzione, di solito è meglio un istogramma
def display_corr_with_col(df, col):
    correlation_matrix = data.corr()
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0, len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x_values, y_values)
    ax.set_title('Correlazione delle features con {}' .format(col), fontsize=20)
    ax.set_ylabel('Coefficiente di correlazione di Pearson', fontsize=16)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()
    
display_corr_with_col(data, 'pep')
# Con il grafico a barre ritrovo gli stessi risultati della matrice di correlazione,
#ma una variabile di correlazione in più: married.
# I risultati dell'analisi ci fanno capire che generalmente la decisione di acquistare o meno il
# PEP dipende da married, children e income, 
# che a sua volta è fortemente influenzato dai fattori children e married.












# CLASSIFICAZIONE E VALIDAZIONE
# Costruzione modello di apprendimento automatico: ora verranno formati diversi modelli di Machine Learning e verranno 
# confrontati i loro risultati.


def get_train_test(df, y_col, x_cols, ratio):
    mask = np.random.rand(len(df)) < ratio 
    df_train = df[mask]
    df_test = df[~mask]
    y_train = df_train[y_col].values
    y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    
    return df_train, df_test, X_train, y_train, X_test, y_test

y_col_glass = 'pep'
x_cols_glass = list(data.columns.values)
x_cols_glass.remove(y_col_glass)

train_test_ratio = 0.7
df_train, df_test, X_train, y_train, X_test, y_test = get_train_test(data, y_col_glass, x_cols_glass, train_test_ratio)

# Creerò un dizionario, che 
# contiene come chiavi il nome dei classificatori e come valori un'istanza dei classificatori.
dict_classifiers = {
    "Nearest Neighbors":  KNeighborsClassifier(),
    "Decision Tree":  tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Naive Bayes": GaussianNB(),
}



def batch_classify(X_train, y_train, X_test, y_test, no_classifiers=5, verbose=True):
    
    dict_models={}
    
    for classifiers_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train,y_train)
        t_end = time.clock()
        
        t_diff =  t_end - t_start
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score':train_score, 'test_score':test_score, 'train_diff':t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c = classifier_name, f=t_diff))
    return dict_models





def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data = np.zeros(shape=(len(cls),4)), columns = ['classifier','train_score','test_score','train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii,'classifier'] = cls[ii]
        df_.loc[ii,'train_score'] = training_s[ii]
        df_.loc[ii,'test_score'] = test_s[ii]
        df_.loc[ii,'train_time'] = training_t[ii]
                       
    display(df_.sort_values(by=sort_by, ascending=False))







# VALUTAZIONE DEI CLASSIFICATORI
# Dopo la definizione dei singoli classificatori è stata fatta una valutazione degli stessi al fine di individuarne 
# il migliore.

# CLASSIFICATORE K-NEIGHBORS
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)

# CLASSIFICATORE DECISION TREE
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

# CLASSIFICATORE RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=1000)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

# CLASSIFICATORE GAUSSIAN (NAIVE BAYES)
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

results = pd.DataFrame({
    
    'Model':['Nearest Neighbors','Decision Tree',
             'Random Forest','Naive Bayes'],
    'Score':[acc_knn, acc_decision_tree,
             acc_random_forest, acc_gaussian]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(8)
print ("\n **********\n Risultati dei diversi classificatori: \n *********\n")
print(results)

# Come è possibile notare i Classificatori Random Forest e Decision Tree hanno una percentutale del 100%, quindi sono 
# quelli più adatti a fare una valutazione. La traccia però ci richiedere di fare una valutazione anche sui 
# classificatoi bayesiani, e confrontare i risultati.







#CROSS VALIDATION
# Ora invece esaminiamo il Classificatore Decision Tree
from sklearn.model_selection import cross_val_score
rf = DecisionTreeClassifier()
scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")
print ("\n **********\n Accuratezza del decision Tree: \n *********\n")
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard deviation: ", scores.std())

# Il modello in questione ha una precisione media del 79% con una deviazione 
# standard del 0.10% circa,che ci mostra, quanto precise sono le stime
#può variare di +/- 0.10%.

# Matrice di Confusione per Decision Tree
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(decision_tree, X_train, y_train, cv=3)
confusion_matrix(y_train, predictions)

# La prima riga della matrice di confusione riguarda le previsioni sui clienti non idonei a possedere un PEP: 169 
# clienti sono stati classificati correttamente come non idonei (veri negativi), mentre 54 sono stati classificati per 
# errore come non idonei (falsi positivi). La seconda riga riguarda, invece, le previsioni sui clienti idonei: 41 che
# per errore sono stati reputati idonei e 144 correttamente classificati come idonei.




#PRECISIONE E RECALL
# Decision Tree
from sklearn.metrics import precision_score, recall_score
print("Precision: ",  precision_score(y_train, predictions))
print("Recall: ", recall_score(y_train, predictions))

# Il modello costruito prevede il 72% delle volte, una classificazione del cliente corretta(precisione)
# La recall afferma che ha predetto la classificazione del 79% di clienti che sono effettivamente idonei.