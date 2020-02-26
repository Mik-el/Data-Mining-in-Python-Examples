import numpy as np
import sklearn as sk
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("CRM.csv", sep=";")
pd.set_option('display.expand_frame_repr', False)

print ("\n **********\n Stampa del n. tot di dati del n. tot di attributi: \n *********\n")
print(data.shape)
print ("\n **********\n Stampa delle prime 5 righe del dataset: \n *********\n")
display(data.head()) 
print ("\n **********\n Stampa delle statistiche descrittive: \n *********\n")
display(data.describe())
print ("\n **********\n Stampa del datatype di ogni colonna: \n *********\n") 
display(data.dtypes)


##############  PREPROCESSING  #######################
# Riempi i dati mancanti con lo 0
data = data.fillna(0)

#######DISCRETIZZAZIONE (valori categorici -> valori numerici) ######

newdata = dict()   #creo un dizionario newdata su cui va
for key in data.keys():
    if data[key].dtypes == 'object':
#dichiaro una LabelEncoder() (trasformerà l'attributo categorico in numerico)
        le = LabelEncoder()
#applico la le
        le.fit(data[key])

        newdata[key] = le.transform(data[key])
    else:
        newdata[key] = data[key]
        
        
#Per applicare l'algoritmo di classificazione 
#copio i dati del dataset da un dizionario a un dataset (nuovo)
res = pd.DataFrame.from_dict(newdata, orient='columns', dtype=None)


###### se first_amount_spent e number_of_products =0 sostituisco la MEDIA ######
numpy_array = res.values
X=numpy_array[:,[1,2]]
#print(X)

imp = SimpleImputer(missing_values=0, strategy="mean")
X = imp.fit_transform(X)
print(X[5])


print ("\n **********\n Stampa delle prime 5 righe del dataset dopo il pre-processing: \n *********\n")
display(res.head())
print ("\n **********\n Stampa delle statistiche descrittive del dataset dopo il preprocessing: \n *********\n")
display(data.describe())





######ANALISI GRAFICA ########

prova = data[['first_amount_spend', 'center']].groupby('age51_89')
prova2 = prova.sum().sort_values('first_amount_spend')
prova2.plot.barh()
print ("\n **********\n Istogramma che rappresenta first_amount_spent in relazione a età 51-89: \n *********\n")
plt.show




############### ANALISI DEI DATI MATRICE DI CORRELAZIONE ####################

data_copy = res.copy(deep=True)
correlation_matrix = data_copy.corr()
plt.figure(figsize=(50,20))

ax = sns.heatmap(correlation_matrix,square=True, annot=True, fmt='.2f', cmap='PuBu', robust=True)

plt.title('Matrice correlazione', fontsize=20)
plt.show()



############### CLASSIFICAZIONE E VALIDAZIONE  ####################
#il metodo trasforma un dataframe in un train e test set, per questo bisogna specificare:
    #1-il ratio train: test (di solito=0.7)
    #2-la colonna con Y_values
def get_train_test(df, y_col, x_cols, ratio):
    
    mask = np.random.rand(len(df)) < ratio 
    df_train = df[mask]
    #~ = Alt + 0126
    df_test = df[~mask]
    
    y_train = df_train[y_col].values
    y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    
    return df_train, df_test, X_train, y_train, X_test, y_test



######################################
y_col_glass = 'Y'
x_cols_glass = list(data_copy.columns.values)
x_cols_glass.remove(y_col_glass)

train_test_ratio = 0.7
df_train, df_test, X_train, y_train, X_test, y_test = get_train_test(data_copy, y_col_glass, x_cols_glass, train_test_ratio)


#######################################
dict_classifiers = {
    "Nearest Neighbors":  KNeighborsClassifier(),
    "Decision Tree":  tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Naive Bayes": GaussianNB(),
}



#######################################
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


#######################################
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



#****** VALUTAZIONE DEI CLASSIFICATORI *******
#accuracy score = knn.score(...)
#CLASSIFICATORE K-NEIGHBORS
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)

#CLASSIFICATORE DECISION TREE
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

#CLASSIFICATORE RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=1000)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

#CLASSIFICATORE GAUSSIAN (NAIVE BAYES)
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
print ("\n **********\n Valutazione classificatori: \n *********\n")
print (results)


############ RANDOM FOREST: CROSS-VALIDATION  ###############
from sklearn.model_selection import cross_val_score
print ("\n **********\n Cross validation/accuratezza Random Forest: \n *********\n")
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard deviation: ", scores.std())


############ PRECISIONE E RECALL  ###############
from sklearn.metrics import precision_score, recall_score

print("Precision: ",  precision_score(y_train, scores))
print("Recall: ", recall_score(y_train, scores))
