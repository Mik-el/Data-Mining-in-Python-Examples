
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

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


data = pd.read_csv("nursery.csv", sep=';')
pd.set_option('display.expand_frame_repr', False)

print ("\n **********\n Stampa delle prime 5 righe del dataset: \n *********\n")
display(data.head()) 
print ("\n **********\n Stampa delle statistiche descrittive: \n *********\n")
display(data.describe())
print ("\n **********\n Numero totale di dati e di attributi: \n *********\n")
print(data.shape)


#PREPROCESSING
#data= data.convert_objects(convert_numeric=True)

# non ci sono dati mancanti. Parò ci sono molti dati 
# categoriali che vanno trasformati in numerici mediante la discretizzazione.
def label_encode(df, columns):
    for col in columns:
        le = LabelEncoder()
        col_values_unique = list(df[col].unique())
        le_fitted = le.fit(col_values_unique)
        
        col_values = list(df[col].values)
        le.classes_
        col_values_transformed = le.transform(col_values)
        df[col] = col_values_transformed
        
data = data.copy(deep=True)
to_be_encoded_cols = data.columns.values
label_encode(data, to_be_encoded_cols)

print ("\n **********\n Stampa delle prime 5 righe del dataset dopo il pre-processing: \n *********\n")
display(data.head()) 
print ("\n **********\n Stampa delle statistiche descrittive dopo il pre-processing: \n *********\n")
display(data.describe())

# Creo un nuovo file csv dove ci copio il dataset processato
f = open("nursery2.csv", "a")
data.to_csv('nursery2.csv', index = False ) #false= non scrive i nomi delle righe
f.close()



#ANALISI GRAFICA
"""print ("\n **********\n Grafico dopo il preprocessing: \n *********\n")
data.plot( kind='barh', x='children', y='has_nurs' )
plt.show()
"""

"""
print ("\n **********\n Stampa istogramma con 30 colonne \n *********\n")
istogramma = data.hist(bins=30)

istogramma.set_title('Relazione tra numero di bambini e presenza infermiera')
istogramma.set_xlabel('Numero letti')
istogramma.set_ylabel('Numero ospedali')

plt.show()
"""



# ANALISI DEI DATI
# Per ottenere maggiori informazioni su come ogni caratteristica è correlata con la Variabile Target, 
# possiamo calcolare e tracciare la matrice di correlazione delle features per il dataset.
# La matrice di correlazione mostra le variabili fortemente correlate con la variabile target, che nel nostro caso
# è la variabile 'class', è 'healt' (condizione di salute).
# uso il metodo .corr
correlation_matrix = data.corr()
plt.figure(figsize=(50,20))
ax = sns.heatmap(correlation_matrix, vmax=1, square=True, 
                 annot=True, fmt='.2f', cmap='GnBu', cbar_kws={"shrink":.5}, robust=True)
plt.title('Matrice di correlazione delle features', fontsize=20)
plt.show()
 

# Una matrice di correlazione è un buon modo per ottenere un quadro generale di come tutte le funzionalità nel set di 
# dati siano correlate tra loro. Per un set di dati con  molte funzionalità potrebbe diventare molto grande e la 
# correlazione di una singola funzione con le altre caratteristiche diventa diﬃcile da discernere.
#Se vogliamo esaminare le correlazioni di una singola funzione, di solito è un'idea migliore visualizzarla sotto forma di grafico 
# a barre.

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
    
display_corr_with_col(data, 'class')

# Con il grafico a barre ritrovo gli stessi risultati della matrice di correlazione.




# CLASSIFICAZIONE E VALIDAZIONE
# Costruzione modello di apprendimento automatico: ora verranno formati diversi modelli di Machine Learning e verranno 
# confrontati i loro risultati.

#Divisione tra Train test e Set Test
def get_train_test(df, y_col, x_cols, ratio):
    """
    Questo metodo trasforma il dataframe in un TRAIN set e in un TEST set, per questo bisogna specificare:
    1. il ratio train: test(in genere 0.7)
    2. la colonna con le Y_values
    """
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]
    
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test

y_col_glass = 'class'
x_cols_glass = list(data.columns.values)
x_cols_glass.remove(y_col_glass)

train_test_ratio = 0.7
df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(data, y_col_glass, x_cols_glass, train_test_ratio)


# Affinchè io possa testare + classificatori, creo un dizionario 
# esso è fatto in modo che CHIAVI= nome dei classificatori
# VALORI =  istanze dei classificatori.
dict_classifier = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(3),
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=10),
    "Decision Tree": tree.DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Naive Bayes": GaussianNB()    
}


def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
    
    dict_models = {} #Scorro il dizionario dei classificatori
    for classifier_name, classifier in list(dict_classifier.items())[:no_classifiers]:
        t_start = time.clock() #Modulo temporale per tenere traccia del tempo necessario ad addestrare il classificatore
        classifier.fit(X_train, Y_train) #Allena il classificatore
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train) #Scelta del classificatore sul set di allenamento
        test_score = classifier.score(X_test, Y_test) #Esegue il classificatore sul set di test
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s" .format(c=classifier_name, f=t_diff))
    return dict_models



#
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
    
# VALUTAZIONE CLASSIFICATORI
# Dopo la definizione dei singoli classificatori è stata fatta una valutazione degli stessi al fine di individuarne 
# il migliore.

# CLASSIFICATORE K-NEIGHBORS
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# CLASSIFICATORE DECISION TREE
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# CLASSIFICATORE RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=1000)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# CLASSIFICATORE GAUSSIAN (NAIVE BAYES)
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

results = pd.DataFrame({
    
    'Model':['Nearest Neighbors','Decision Tree',
             'Random Forest','Naive Bayes'],
    'Score':[acc_knn, acc_decision_tree,
             acc_random_forest, acc_gaussian]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(8)
print (results)
# Come è possibile notare i Classificatori Random Forest e Decision Tree hanno una percentutale del 100%, quindi sono 
# quelli più adatti a fare una valutazione
#??? NON VEDO NESSUN OUTPUT



#CROSS VALIDATION per Random Forest.
#Per valutare il classificatore è stata fatta un’ulteriore verifica 
#applicando una convalida incrociata.
#Il Cross Validation divide casualmente il train set in k sottoinsiemi chiamati folds.
#Se dividessi i dati in 10 folds(K = 10),il modello del classificatore verrebbe 
# addestrato e valutato 10 volte, usando ogni volta 9 fold da train test e 1 fold da test set 
#Pertanto ottengo un array con 10 punteggi diversi.
print ("\n **********\n Risultati cross validation \n *********\n")
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Punteggio: ", scores)
print("Media: ", scores.mean())
print("Deviazione standard: ", scores.std())

# Come si può osservare dall'output il modello usato (Random Forest) ha una precisione media del 94% con una deviazione 
# standard del 0.06%,che ci mostra, quanto precise sono le stime. Ciò significa che la precisione del nostro modello 
# può variare di +/- 0.06%. Dunque, a seguito di questa verifica, la precisione continua ad essere ancora buona per cui 
# nelle fasi successive si proverà a migliorare ulteriormente le prestazioni del Random Forest.


# Matrice di Confusione per Random Forest
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)

# PRECISIONE E RECALL (per il random forest)
from sklearn.metrics import precision_score, recall_score
print("Precision: ",  precision_score(Y_train, predictions))
print("Recall: ", recall_score(Y_train, predictions))


"""
# Ora invece CROSS VALIDATION per Decision Tree
from sklearn.model_selection import cross_val_score
rf = DecisionTreeClassifier()
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard deviation: ", scores.std())

# Abbiamo all'incirca gli stessi risultati del Random Forest per quanto riguarda la deviazione standard (0.04%), mentre
# la precisione media è del 96%, quindi molto più accurato del Random Forest.


# Matrice di Confusione per Decision Tree
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(decision_tree, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)

# PRECISION E RECALL
from sklearn.metrics import precision_score, recall_score
print("Precision: ",  precision_score(Y_train, predictions))
print("Recall: ", recall_score(Y_train, predictions))
"""