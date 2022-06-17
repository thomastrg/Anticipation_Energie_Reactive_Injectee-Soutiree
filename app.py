"""
@author: thomas TRANG
"""
#Basics
from ensurepip import bootstrap
import numpy as np
import pandas as pd
#Déploiement    
import hydralit_components as hc
import hydralit as hy
import streamlit as st 

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor





app = hy.HydraApp(title="Anticipation énergie réactive | Enedis ",hide_streamlit_markers=True,layout='wide')



@app.addapp(title='Prédiction ER injectée')
def app1():
    #Présentation
    st.title("Cette application prédit l'arrivée d'énergie réactive")
    st.subheader("Sur cet onglet, on prédira l'arrivée d'énergie réactive injectée")    
    st.write("La modèle de machine learning adopté est le Random Forest regressor")
    st.write("Il a un score R2 de 0.59 après tuning")    
    st.write('---')
    
    
    #Side Bar
    st.sidebar.title("Facteurs")
    st.sidebar.title("Saisissez les valeurs des facteurs :")   
    long_hta = float(st.sidebar.text_input("Longueur HTA souterraine (en km):", "100"))
    prod_eolien = float(st.sidebar.text_input("Production éolienne (en kW):", "1000"))
    prod_photo = float(st.sidebar.text_input("Production photovoltaïque (en kW):", "1000"))
    prod_autre = float(st.sidebar.text_input("Production autres EnR (en kW):", "1000"))
    
    #Retrieve User input
    data ={ 'Longueur HTA' : long_hta,
           'Production eolienne' : prod_eolien,
           'Production photovoltaïque' : prod_photo,
           'Production autres EnR' : prod_autre,
    }
    st.subheader("Voici les facteurs de la prédiction ci-dessous")
    values_to_pred=pd.DataFrame(data,index=[0])
    st.dataframe(values_to_pred)
    
    
    #Import data 
    df_init=pd.read_excel('dataset_final.xlsx',index_col=0,engine='openpyxl')
    df_init=df_init[df_init.columns[:-1]]
    df=df_init[df_init.columns[2:]]
    df_entrainement=df
    
    
    # modeling 
    X=df_entrainement[df_entrainement.columns[:-2]]
    y_inj=df_entrainement[df_entrainement.columns[-2]]
    
    X_train,X_test,y_train,y_test=train_test_split(X,y_inj,test_size=0.27)
    
    # Random Forest regressor 
    # Parameters with n estimators = 100.00, max depth : 80.00, min samples split : 2 and min samples leaf : 1 and bootstrap : True  
    randomForestAlgo = RandomForestRegressor(n_estimators=100, max_depth=80, min_samples_split=2, min_samples_leaf=1, bootstrap=True)
    randomForestAlgo.fit(X_train,y_train)
    pred=randomForestAlgo.predict(values_to_pred)
    st.write('---')
    st.write('')
    st.subheader("La prédiction du modèle d'énergie réactive injectée produite est de "+str("{:,}".format(pred[0])).replace(',',' ')+' kWh')
    



@app.addapp(title='Prédiction ER soutirée')
def app1():    
    st.title("Cette application prédit l'arrivée d'énergie réactive") 
    st.subheader("Sur cet onglet, on prédira l'arrivée d'énergie réactive soutirée")    
    st.write("La modèle de machine learning adopté est le Extras Trees regressor")
    st.write("Il a un score R2 de 0.52 après tuning")    
    st.write('---')
    
    
    #Side Bar
    st.sidebar.title("Facteurs")   
    st.sidebar.title("Saisissez les valeurs des facteurs :")   
    long_hta = float(st.sidebar.text_input("Longueur HTA souterraine (en km):", "100"))
    prod_eolien = float(st.sidebar.text_input("Production éolienne (en kW):", "1000"))
    prod_photo = float(st.sidebar.text_input("Production photovoltaïque (en kW):", "1000"))
    prod_autre = float(st.sidebar.text_input("Production autres EnR (en kW):", "1000"))
    
    #Retrieve User input
    data ={ 'Longueur HTA' : long_hta,
           'Production eolienne' : prod_eolien,
           'Production photovoltaïque' : prod_photo,
           'Production autres EnR' : prod_autre,
    }
    st.subheader("Voici les facteurs de la prédiction ci-dessous")
    values_to_pred=pd.DataFrame(data,index=[0])
    st.dataframe(values_to_pred)
    
    
    #Import data     
    df_init=pd.read_excel('dataset_final.xlsx',index_col=0,engine='openpyxl')
    df_init=df_init[df_init.columns[:-1]]
    df=df_init[df_init.columns[2:]]
    df_entrainement=df

    
    # modeling 
    X=df_entrainement[df_entrainement.columns[:-2]]
    y_sout=df_entrainement[df_entrainement.columns[-1]]

    
    X_train,X_test,y_train,y_test=train_test_split(X,y_sout,test_size=0.27)
    
                                                   
    # Random Forest regressor 
    # Parameters with n estimators = 100.00, max depth : 80.00, min samples split : 2 and min samples leaf : 1 and bootstrap : True  
    extraTreesAlgo = ExtraTreesRegressor(n_estimators=90, max_depth=80, min_samples_split=2, min_samples_leaf=1, bootstrap=True)
    extraTreesAlgo.fit(X_train,y_train)
    pred=extraTreesAlgo.predict(values_to_pred)
    st.write('---')
    st.write('')
    st.subheader("La prédiction du modèle d'énergie réactive soutirée produite est de "+str("{:,}".format(pred[0])).replace(',',' ')+' kWh')             
    
    
app.run()