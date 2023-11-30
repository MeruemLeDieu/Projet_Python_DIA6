"""
Created on Sun Nov 19 16:49:19 2023

@author: Subina Suthan
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error




df = pd.read_csv("online_shoppers_intention.csv")
df2 = df.copy()


st.title('Analyse du comportement en ligne des visiteurs')

st.dataframe(df.head())

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Traduction des données", "Analyse de données",
         "Exploration des données", "Modélisation des données", "Deep Learning"]

page = st.sidebar.radio("Aller vers la page :", pages)

# Encodage de la colonne "Month"
df2['Month'] = df2['Month'].map({'Feb': 0, 'Mar': 1, 'May': 2, 'June': 3, 'Jul': 4, 'Aug': 5, 'Sep': 6, 'Oct': 7, 'Nov': 8, 'Dec': 9})
df2['Month'] = df2['Month'].astype(int)

# Encodage de la colonne "VisitorType"
df2['VisitorType'] = df2['VisitorType'].map({'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2})
df2['VisitorType'] = df2['VisitorType'].astype(int)

# Conversion de la colonne "Weekend" en 0 et 1
df2['Weekend'] = df2['Weekend'].astype(int)

# Conversion de la colonne "Revenue" en 0 et 1
df2['Revenue'] = df2['Revenue'].astype(int) 

if page == pages[0]:
    st.write("### Contexte du projet")
    st.write("Ce projet vise à analyser le comportement en ligne des visiteurs sur un site web commercial. Les données proviennent de sessions de visite, chacune caractérisée par divers attributs tels que le nombre de pages administratives, le temps passé sur des pages de produits, les taux de rebond, ...")
    st.write("L'objectif est de comprendre les tendances qui peuvent influencer les clients. L'analyse inclut une exploration des données, des visualisations des données, et aussi d'autres aspects.")
    st.write("N'hésitez pas à explorer les différentes pages de l'application pour obtenir plus de détails.")

if page == pages[1]:
    st.write("### Traduction des catégories")

    st.write("Le jeu de données contient plusieurs types de données, notamment :")
    st.write(df.dtypes)

    st.write("#### Régions:")
    st.write("Les régions ont été encodées avec des numéros. Voici la correspondance avec les noms de régions :")
    st.write(pd.DataFrame({'Code': ['1', '2', '3', '4', '5', '6', '7', '8', '9'], 'Région': [
             'Malaisie', 'Italie', 'Espagne', 'Allemagne', 'Angleterre', 'Pologne', 'Colombie', 'Roumanie', 'France']}))

    st.write("#### Navigateurs:")
    st.write("Les navigateurs ont également été encodés avec des numéros. Voici la correspondance avec les noms de navigateurs :")
    st.write(pd.DataFrame({'Code': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'], 'Navigateur': ['Chrome', 'Duckduckgo', 'Mozilla Firefox',
             'Microsoft Edge', 'Safari', 'Vivaldi', 'Opera', 'TOR Browser', 'Maxthon', 'Torch Browser', 'UC Browser', 'Avast Secure Browser', 'Chromium Browser']}))

    st.write("#### Systèmes d'exploitation:")
    st.write("Les systèmes d'exploitation ont été encodés avec des numéros. Voici la correspondance avec les noms de systèmes d'exploitation :")
    st.write(pd.DataFrame({'Code': ['1', '2', '3', '4', '5', '6', '7', '8'], 'Système d\'exploitation': [
             'Windows', 'MacOS', 'IOS', 'Android', 'GNU', 'Linux', 'Unix', 'RTX']}))




if page == pages[2]:
    st.write("## Analyse des données")

    correlation_matrix = df2.corr()

    st.write("### Corrélation entre les variables numériques")
    st.write(px.imshow(correlation_matrix,
             color_continuous_scale='viridis', labels=dict(color='Corrélation')))

    st.write('### Total du temps passé par un visiteur par type de pages')
    coucou = df[['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated',
                'ProductRelated_Duration']]
    fig_total_time = plt.figure()
    nb_total_time_per_type_of_page = coucou.sum()
    nb_total_time_per_type_of_page.plot(kind='bar')
    plt.title("Total du temps passé par un visiteur par type de pages")
    plt.ylabel("Temps en secondes")
    st.pyplot(fig_total_time)

    img = 'moyenne.png'
    st.write("### Moyenne du temps passé par un visiteur par type de pages")
    st.image(img, caption=" ")

    st.write("### Somme et moyenne temps passé par un visiteur par type de page")
    fig, ax1 = plt.subplots()
    nb_total_time_per_type_of_page = coucou.sum()
    mean_time_per_type_of_page = coucou.mean()
    ax1.bar(nb_total_time_per_type_of_page.index, nb_total_time_per_type_of_page.values,
            alpha=0.5, color='blue', label='Somme des transactions')
    ax1.set_ylabel('Total du temps passé par un visiteur par type de page')
    ax1.tick_params(labelcolor='blue')
    ax1.set_ylim([0, max(nb_total_time_per_type_of_page.values) * 1.1])
    ax2 = ax1.twinx()
    ax2.bar(mean_time_per_type_of_page.index, mean_time_per_type_of_page.values,
            alpha=0.5, color='red', label='Moyenne des transactions')
    ax2.set_ylabel('Moyenne temps passé par un visiteur par type de page')
    ax2.tick_params(labelcolor='red')
    ax2.set_ylim([0, max(mean_time_per_type_of_page.values) * 1.3])
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_xticklabels(coucou.columns, rotation=90,
                        ha='right', color='black', fontsize=5)
    plt.title('Somme et moyenne temps passé par un visiteur par type de page')
    st.pyplot(fig)

    ordre = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    st.write('### Temps de visite total sur une page de type administrative par mois')
    Administrative = df[['Administrative', 'Month']]
    Administrative = Administrative.groupby('Month').Administrative.sum().reindex(index=ordre, fill_value=0)

    fig, ax = plt.subplots()
    ax.plot(Administrative.index, Administrative.values, marker='o')
    ax.set_xlabel('Mois')
    ax.set_ylabel('Temps de visite total sur une page de type administrative')
    ax.set_title(
        'Temps de visite total sur une page de type administrative par mois')
    ax.grid(True)
    st.pyplot(fig)

    st.write('### Temps de visite total sur une page de type Administrative_Duration par mois')
    Administrative_duration = df[['Administrative_Duration', 'Month']]
    Administrative_duration = Administrative_duration.groupby('Month').Administrative_Duration.sum().reindex(index=ordre, fill_value=0)
    fig_Administrative_duration, ax = plt.subplots() 
    ax.plot(Administrative_duration.index, Administrative_duration.values, marker='o')
    ax.set_xlabel('Mois')
    ax.set_ylabel('Temps de visite total sur une page de type Administrative_Duration')
    ax.set_title('Temps de visite total sur une page de type Administrative_Duration par mois')
    ax.grid(True)
    st.pyplot(fig_Administrative_duration)

    st.write('### Temps de visite total sur une page de type Informational par mois')
    Informational = df[['Informational', 'Month']]
    Informational = Informational.groupby('Month').Informational.sum().reindex(index=ordre, fill_value=0)
    fig_info = plt.figure()
    plt.plot(Informational.index, Informational.values, marker='o')
    plt.xlabel('Mois')
    plt.ylabel('Temps de visite total sur une page de type Informational')
    plt.title('Temps de visite total sur une page de type Informational par mois')
    plt.grid(True)
    st.pyplot(fig_info)
    
    st.write('### Temps de visite total sur une page de type Informational_Duration par mois')
    Informational_Duration = df[['Informational_Duration', 'Month']]
    Informational_Duration = Informational_Duration.groupby('Month').Informational_Duration.sum().reindex(index=ordre, fill_value=0)
    fig_info_duration = plt.figure()
    plt.plot(Informational_Duration.index, Informational_Duration.values, marker='o')
    plt.xlabel('Mois')
    plt.ylabel('Temps de visite total sur une page de type Informational_Duration')
    plt.title('Temps de visite total sur une page de type Informational_Duration par mois')
    plt.grid(True)
    st.pyplot(fig_info_duration)


    st.write('### Temps de visite total sur une page de type ProductRelated par mois')
    ProductRelated = df[['ProductRelated', 'Month']]
    ProductRelated = ProductRelated.groupby('Month').ProductRelated.sum().reindex(index=ordre, fill_value=0)
    fig_product_related = plt.figure()
    plt.plot(ProductRelated.index, ProductRelated.values, marker='o')
    plt.xlabel('Mois')
    plt.ylabel('Temps de visite total sur une page de type ProductRelated')
    plt.title('Temps de visite total sur une page de type ProductRelated par mois')
    plt.grid(True)
    st.pyplot(fig_product_related)


    st.write('### Temps de visite total sur une page de type ProductRelated_Duration par mois')
    ProductRelated_Duration = df[['ProductRelated_Duration', 'Month']]
    ProductRelated_Duration = ProductRelated_Duration.groupby('Month').ProductRelated_Duration.sum().reindex(index=ordre, fill_value=0)
    fig_product_related_duration = plt.figure()
    plt.plot(ProductRelated_Duration.index, ProductRelated_Duration.values, marker='o')
    plt.xlabel('Mois')
    plt.ylabel('Temps de visite total sur une page de type ProductRelated_Duration')
    plt.title('Temps de visite total sur une page de type ProductRelated_Duration par mois')
    plt.grid(True)
    st.pyplot(fig_product_related_duration)

    st.write('### Visualisation combinée pour toutes les pages de type')
    fig_combined = plt.figure()   
    plt.plot(Administrative.index, Administrative.values, label='Administrative')
    plt.plot(Administrative_duration.index, Administrative_duration.values, label='Administrative_Duration')
    plt.plot(Informational.index, Informational.values, label='Informational')
    plt.plot(Informational_Duration.index, Informational_Duration.values, label='Informational_Duration')
    plt.plot(ProductRelated.index, ProductRelated.values, label='ProductRelated')
    plt.plot(ProductRelated_Duration.index, ProductRelated_Duration.values, label='ProductRelated_Duration')
    plt.xlabel('Mois')
    plt.ylabel('Temps de visite total sur une page de type')
    plt.title('Temps de visite total sur une page de type par mois')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_combined)


    st.write('### Visualisation de la répartition des visites en fonction des mois')
    all_months_df = pd.DataFrame(index=ordre)
    merged_df = pd.merge(all_months_df, df, left_index=True, right_on='Month', how='left')
    st.title('Répartition des visites en fonction des mois')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Month', hue='Month', data=merged_df, palette='muted', order=ordre, ax=ax)
    plt.title('Répartition des visites en fonction des mois')
    st.pyplot(fig)

    st.write('### Visualisation de la répartition des Operating Systems')
    fig_os_repartition = plt.figure(figsize=(10, 6))
    sns.countplot(x='OperatingSystems', hue='OperatingSystems',
                  data=df, palette='viridis')
    plt.title('Répartition des Operating Systems')
    st.pyplot(fig_os_repartition)

    region_distribution = df['Region'].value_counts().reset_index()
    region_distribution.columns = ['Region', 'Count']
    colors = {'Malaisie': 'lightblue', 'France': 'lightgreen', 'Espagne': 'lightcoral', 'Italie': 'lightsalmon',
              'Allemagne': 'lightseagreen', 'Pologne': 'lightpink', 'Colombie': 'lightyellow', 'Roumanie': 'lightcyan', 'Angleterre': 'lightgrey'}
    fig = make_subplots(rows=1, cols=1, subplot_titles=[
                        'Répartition des Régions'])
    for i, row in region_distribution.iterrows():
        fig.add_trace(go.Bar(x=[row['Region']], y=[row['Count']], marker=dict(
            color=colors.get(row['Region'], 'lightgray'))), row=1, col=1)
    fig.update_layout(title_text='Répartition des Régions', showlegend=False, height=500, width=800,
                      xaxis=dict(title='Région'), yaxis=dict(title='Nombre de sessions'))
    st.plotly_chart(fig)

    st.write('### Visualisation des Browsers')
    browser_distribution = df['Browser'].value_counts().reset_index()
    browser_distribution.columns = ['Browser', 'Count']
    threshold = 170
    browser_distribution['Browser'] = browser_distribution['Browser'].where(
        browser_distribution['Count'] >= threshold, 'Autre')
    browser_distribution = browser_distribution.groupby(
        'Browser', as_index=False)['Count'].sum()
    max_index = browser_distribution['Count'].idxmax()
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=browser_distribution['Browser'], values=browser_distribution['Count'], hole=0.3, textinfo='percent+label', marker=dict(
        line=dict(color='#000000', width=1.5)), pull=[0.1 if i == max_index else 0 for i in range(len(browser_distribution))]))
    fig.update_layout(title='Répartition des Browsers', showlegend=True, margin=dict(l=50, r=50, b=100, t=100), height=500, legend=dict(
        x=1, y=1, traceorder='normal', font=dict(size=12), bgcolor='#E2E2E2', bordercolor='#FFFFFF', borderwidth=2))
    st.plotly_chart(fig)

    st.write('### Visualisation de la répartition des TrafficTypes')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='TrafficType', hue='TrafficType', data=df, palette='Set1')
    plt.title('Répartition des TrafficTypes')
    fig_traffic_types = plt.gcf()  # Get the current figure
    st.pyplot(fig_traffic_types)

    img = "pairplot.png"
    st.write('### Pairplot des variables numériques avec Revenue en couleur')
    st.image(img, caption="Pairplot des variables numériques avec Revenue en couleur",
             use_column_width=True)
    # fig = plt.figure(figsize=(10, 8))  # Create a figure object
    #sns.pairplot(df, hue='Revenue', diag_kind='kde')
    #plt.suptitle('Pairplot des variables numériques avec Revenue en couleur')
    # st.pyplot(fig)

    st.write('### Boxplots des pages visitées par type en fonction du Revenue')
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    for i, col in enumerate(['Administrative', 'Informational', 'ProductRelated']):
        sns.boxplot(x='Revenue', y=col, data=df, ax=axes[i])
        axes[i].set_title(f'Boxplot de {col}')
    plt.tight_layout()
    plt.suptitle(
        'Boxplots des pages visitées par type en fonction du Revenue', y=1.02)
    st.pyplot(fig)

    st.write('### Répartition des types de visiteurs en fonction du week-end')
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.countplot(x='VisitorType', hue='Weekend', data=df, ax=ax1)
    ax1.set_title('Répartition des types de visiteurs en fonction du week-end')
    st.pyplot(fig1)

    st.write('### Distribution des visites en fonction du mois')
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.histplot(x='Month', hue='Revenue', multiple='stack', data=df, ax=ax2)
    ax2.set_title('Distribution des visites en fonction du mois')
    st.pyplot(fig2)

    st.write('### Histogrammes des variables par classe de revenu')
    df_revenue_true = df[df['Revenue'] == True]
    df_revenue_false = df[df['Revenue'] == False]
    for col in df.columns[:-1]:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.histplot(df_revenue_true[col], color='blue',
                     label='Revenue=True', kde=True, ax=ax3)
        sns.histplot(df_revenue_false[col], color='orange',
                     label='Revenue=False', kde=True, ax=ax3)
        ax3.set_title(f'Histogramme de {col} par classe de revenu')
        ax3.legend()
        st.pyplot(fig3)

    st.write("##### Prenons le log pour mettre les données dans une bonne échelle pour faire de meilleures comparaisons.")
    st.image('log1.png', caption=" ")
    st.image('log2.png', caption=" ")
    st.image('log3.png', caption=" ")
    st.image('log4.png', caption=" ")
    st.image('log5.png', caption=" ")
    st.image('log6.png', caption=" ")
    st.image('log7.png', caption=" ")
    st.image('log8.png', caption=" ")
    st.image('log9.png', caption=" ")
    st.image('log10.png', caption=" ")
    st.image('log11.png', caption=" ")
    st.image('log12.png', caption=" ")
    st.image('log13.png', caption=" ")
    st.image('log14.png', caption=" ")
    st.image('log15.png', caption=" ")
    st.image('log16.png', caption=" ")
    st.image('log17.png', caption=" ")

    # st.write("### Violinplot de la distribution des revenus par mois")
    #fig4, ax4 = plt.subplots(figsize=(14, 8))
    #sns.violinplot(x='Month', y='Revenue', hue='Month', data=df, palette='viridis', ax=ax4)
    #ax4.set_title('Violinplot de la distribution des revenus par mois')
    # st.pyplot(fig4)

    st.write('### Violinplots des durées de visite en fonction du Revenue')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    for i, col in enumerate(['Administrative_Duration', 'ProductRelated_Duration']):
        sns.violinplot(x='Revenue', y=col, data=df, ax=axes[i])
    plt.suptitle('Violinplots des durées de visite en fonction du Revenue')
    st.pyplot(fig)

    #fig, ax = plt.subplots(figsize=(14, 8))
    #sns.violinplot(x='Month', y='Revenue', hue='Month', data=df, palette='viridis', ax=ax)
    #plt.title('Violinplot de la distribution des revenus par mois')
    # st.pyplot(fig)

    image_stream = "violinplot2.png"
    st.image(image_stream, caption='Violinplot de la distribution des revenus par mois',
             use_column_width=True)

    st.write('### Répartition des types de visiteurs par classe de revenu')
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.countplot(x='VisitorType', hue='Revenue',
                  data=df, palette='pastel', ax=ax5)
    ax5.set_title('Répartition des types de visiteurs par classe de revenu')
    st.pyplot(fig5)

    st.write("### Répartition des systèmes d'exploitation par classe de revenu")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.countplot(x='OperatingSystems', hue='Revenue',
                  data=df, palette='Set2', ax=ax6)
    ax6.set_title(
        "Répartition des systèmes d'exploitation par classe de revenu")
    st.pyplot(fig6)


if page == pages[3]:
    st.write("### Exploration des données")
    st.write("##### SpecialDay")
    specific_day_visits = df[df['SpecialDay'] == 1]
    num_visits_specific_day = len(specific_day_visits)
    st.write(
        f"Le nombre de personnes qui ont visité le site ce jour-là était de {num_visits_specific_day}.")
    st.write("##### Browser")
    st.write("Le navigateur le plus utilisé est DuckDuckGo.")
    st.write("##### Month")
    st.write(
        "Le mois de mai enregistre le plus grand nombre de visites, suivi de novembre.")
    st.write("Novembre est le mois où le plus grand nombre de pages a été consulté.")
    st.write("##### Operating System")
    st.write("MacOS est l'OS le plus utilisé par les visiteurs.")
    st.write("##### Country")
    st.write(
        "La Malaisie est le pays où le plus grand nombre de commandes a été effectué.")
    st.write("##### Traffic Type")
    st.write("Le Traffic Type 2 est le plus utilisé.")
    st.write("##### Visitor Type")
    st.write(
        "Les jours de week-end ont plus de visiteurs déjà connus que de nouveaux visiteurs.")


# désolée monsieur mais ça prenait trop de temps à run
if page == pages[4]:
    df2=df.copy()
    st.write("# Classification")
    
    st.write("### RandomForestClassifier avec PCA")
    st.image('rfcavecpca.png', caption=" ")
    st.image('rfcavecpca2.png', caption=" ")
    st.write("### RandomForestClassifier sans PCA")
    st.image('rfcsanspca.png', caption=" ")
    
    st.write("### KNeighborsClassifier avec PCA")
    st.image('knnavecpca.png', caption=" ")
    st.image('knnavecpca2.png', caption=" ")
    st.write("### KNeighborsClassifier sans PCA")
    st.image('knnsanspca.png', caption=" ")
    
    st.write("### GaussianNB avec PCA")
    st.image('gaussiannbavecpca.png', caption=" ")
    st.image('gaussiannbavecpca2.png', caption=" ")
    st.write("### GaussianNB sans PCA")
    st.image('gaussiannbsanspca.png', caption=" ")
    
    st.write("### SVC avec PCA")
    st.image('svcavecpca.png', caption=" ")
    st.image('svcavecpca2.png', caption=" ")
    st.write("### SVC sans PCA")
    st.image('svcsanspca.png', caption=" ")
    
    st.write("### XGBClassifier avec PCA")
    st.image('xgbclassifier1.png', caption=" ")
    st.image('xgbclassifier2.png', caption=" ")
    st.write("### XGBClassifier sans PCA")
    st.image('xgbclassifier3.png', caption=" ")
    st.image('xgbclassifier4.png', caption=" ")
    
    
    st.write("### Regression Logistic sans PCA")
    st.image('regressionlogistic.png', caption="")
    
#    target = df2['Revenue']
#    data_for_classification = df2.copy().drop(columns='Revenue')
#    numeric_cols = data_for_classification.select_dtypes(include=['number']).columns
#    non_numeric_cols = data_for_classification.select_dtypes(exclude=['number']).columns
#    scaler = StandardScaler()
#    pca = PCA(n_components=2)
#    data_for_classification_numeric_scaled = scaler.fit_transform(data_for_classification[numeric_cols])
#    data_for_classification_pca = pca.fit_transform(data_for_classification_numeric_scaled)
#    data_for_classification_final = pd.concat([pd.DataFrame(data_for_classification_pca, columns=['PCA1', 'PCA2']),
#                                               data_for_classification[non_numeric_cols]], axis=1)
#    X_train, X_test, y_train, y_test = train_test_split(data_for_classification_pca, target, test_size=0.2, random_state=42)
#    X_train2, X_test2, y_train2, y_test2 = train_test_split(data_for_classification_numeric_scaled, target, test_size=0.2, random_state=42)
#    pipel_LR = Pipeline(steps=[('scale', StandardScaler()), ('model', LogisticRegression())])
#    pipel_LR.fit(X_train2, y_train2)
#    y_pred = pipel_LR.predict(X_test2)
#    st.title("Regression Logistic sans PCA")
#    st.subheader("Prédicition des lignes:")
#    st.write(y_pred)
#    accuracy = accuracy_score(y_test2, y_pred)
#    confusion = confusion_matrix(y_test2, y_pred)
#    classification_rep = classification_report(y_test2, y_pred)    
#    st.subheader("Métriques:")
#    st.write(f"Exactitude: {accuracy}")
#    st.write(f"Matrice de confusion:\n{confusion}")
#    st.write(f"Rapport de classification:\n{classification_rep}")
#    st.subheader("Conclusion:")
#    if accuracy > 0.8:
#        st.write("Très bonne précision pour de la régression.")
#    else:
#        st.write("Mauvaise précision.")





if page == pages[5]:
    st.write("# Deep Learning")
    st.write("## Qu'est-ce que le Deep Learning ?")
    st.write("##### Le Deep Learning, autrement l'apprentissage profond en français, est un procédé d'apprentissage automatique utilisant des réseaux de neurones possédants plusieurs couches de neurones cachées. Ces algorithmes possédant de très nombreux paramètres, ils demandent un nombre très important de données afin d'être entraînés. ")

    st.write("## Deep Learning avec PCA")
    img = 'deeplearning.png'
    img2 = 'deeplearning2.png'
    st.image(img, caption="")
    st.image(img2, caption=" ")
    st.image('deeplearning5.png', caption=" ")

    
    st.write("## Deep Learning sans PCA")
    img = 'deeplearning3.png'
    st.image(img, caption="")
    st.image('deeplearning6.png', caption=" ")

    
    st.write("### Conclusion")
    st.write("### 1... 2... 3...")
    st.write("##### Notre réseau de neurones sans PCA remporte la bataille avec :")
    st.image('deeplearning7.png', caption="")
    
    


#    df2=df.copy()

#    plt.style.use('dark_background')
#    plt.rcParams.update({
#        "figure.facecolor": (0.12, 0.12, 0.12, 1),
#        "axes.facecolor": (0.12, 0.12, 0.12, 1),
#    })
#    numeric_columns = df2.select_dtypes(include=['int64', 'float64']).columns
#    data_for_classification = df2[numeric_columns]
#    scaler = StandardScaler()
#    data_for_classification_scaled = scaler.fit_transform(data_for_classification)
#    pca = PCA(n_components=2)
#    data_for_classification_pca = pca.fit_transform(data_for_classification_scaled)
#    target = df2['Revenue']

#    X = data_for_classification_pca.T
#    X = X.astype(np.float64)
#    y = target.to_numpy().reshape((1, target.shape[0]))
#    y = np.where(y == True, 1, 0)
#    print("X = ", X)
#    print(X.shape)
#    print(type(X))
#    print("y = ", y)
#    print(y.shape)
#    print(type(y))
#    def Initialisation(dimensions):
#        parametres = {}
#        C = len(dimensions)

#        for c in range(1, C):
#            parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
#            parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

#        return parametres
#    def Predict(X, parametres):
#        activations = Forward_Propagation(X, parametres)
#        C = len(parametres) // 2
#        Af = activations['A' + str(C)]
#        return Af >= 0.5
#    def visualisation(X, y, params):
#        fig, ax = plt.subplots()
#        ax.scatter(X[0, :], X[1, :], c=y, cmap='bwr', s=50)
#        x0_lim = ax.get_xlim()
#        x1_lim = ax.get_ylim()

#        resolution = 100
#        x0 = np.linspace(x0_lim[0], x0_lim[1], resolution)
#        x1 = np.linspace(x1_lim[0], x1_lim[1], resolution)

    # meshgrid
#        X0, X1 = np.meshgrid(x0, x1)

    # assemble (100, 100) -> (10000, 2)
#        XX = np.vstack((X0.ravel(), X1.ravel()))

#        Z = Predict(XX, params)
#        Z = Z.reshape((resolution, resolution))

#        ax.pcolormesh(X0, X1, Z, cmap='bwr', alpha=0.3, zorder=-1)
#        ax.contour(X0, X1, Z, colors='green')

#        plt.show()
#    def Forward_Propagation(X, parametres):
#        activations = {'A0' : X}
#        C = len(parametres) // 2

#        for c in range(1, C + 1):
#            Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
#            activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

#        return activations
#    def Back_Propagation(y, activations, parametres):

#        m = y.shape[1]
#        C = len(parametres) // 2

#        dZ = activations['A' + str(C)] - y
#        gradients = {}

#        for c in reversed(range(1, C + 1)):
#            gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
#            gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
#            if (c > 1):
#                dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

#        return gradients
#    def Update(gradients, parametres, learning_rate):
#        C = len(parametres) // 2

#        for c in range(1, C + 1):
#            parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
#            parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

#        return parametres
#    def Neural_Network(X, y, hidden_layers = (32, 32, 32), learning_rate = 0.1, n_iter = 1000):
    # Initialisation W, b
#        dimensions = list(hidden_layers)
#        dimensions.insert(0, X.shape[0])
#        dimensions.append(y.shape[0])
#        np.random.seed(1)
#        parametres = Initialisation(dimensions)
    # tableau numpy contenant les futures accuracy et log_loss
#        training_history = np.zeros((int(n_iter), 2))

#        C = len(parametres) // 2

#        for i in tqdm(range(n_iter)):
#            activations = Forward_Propagation(X, parametres)
#            gradients = Back_Propagation(y, activations, parametres)
#            parametres = Update(gradients, parametres, learning_rate)
#            Af = activations['A' + str(C)]

    # calcul du log_loss et de l'accuracy
#            training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
#            y_pred = Predict(X, parametres)
#            training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))

    # Plot courbe d'apprentissage
#        plt.figure(figsize=(12, 4))
#        plt.subplot(1, 2, 1)
#        plt.plot(training_history[:, 0], label='train loss')
#        plt.legend()
#        plt.subplot(1, 2, 2)
#        plt.plot(training_history[:, 1], label='train acc')
#        plt.legend()
#        visualisation(X, y, parametres)
#        plt.show()

#        return training_history
#    training_history = Neural_Network(X, y, hidden_layers=(64, 64, 64, 64, 64, 64, 64, 64), learning_rate=0.001, n_iter=10000)

#    st.write("## Historique d'entraînement")
#    plt.plot(training_history)
#    plt.xlabel('Itération')
#    plt.ylabel('Perte')
#    st.pyplot(plt)
