import streamlit as st
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

#st.dataframe(df.style.highlight_max(axis=0))
#df.head(5)
st.title('Case Portocred Financeira')
st.file_uploader("Subir arquivos",accept_multiple_files=True)

st.subheader("Desenvolvido por: Renato Dias")
st.sidebar.title("Exploração de dados (EAD)")

st.sidebar.subheader("Visões gerais")
col1, col2,col3 = st.sidebar.beta_columns(3)
pr = col1.checkbox('Stats', False)
bx = col2.checkbox('Boxplot', False)
cor = col3.checkbox("Correlação", False)

st.sidebar.subheader("Tratamento de outliers")
chosen = st.sidebar.radio('',("Sim","Não"))

st.sidebar.subheader("Balanceamento das classes")
c1 = st.sidebar.checkbox("Verificar percentual por classe")
st.sidebar.text("Balancear classes?")
col11, col21 = st.sidebar.beta_columns(2)
chosen1 = col11.radio('',(" Sim"," Não"))

st.sidebar.subheader("Tratamento dos dados")
chosen2 = st.sidebar.radio('',("  Sim","  Não"))

st.sidebar.subheader("Padronização/Normalização")
chosen3 = st.sidebar.radio('',("   Sim","   Não"))


st.sidebar.subheader("Categorização de variáveis")
chosen4 = st.sidebar.radio(' ',("Sim","Não"))


if chosen4 == "Sim":
    st.sidebar.slider("Bins",2,10)

st.sidebar.subheader("Indicadores")
iv = st.sidebar.checkbox("IV", False)

st.sidebar.subheader("Ponto de corte (cutoff)")
chosen5 = st.sidebar.radio('  ',("Sim","Não"))

if chosen5 == "Sim":
    add_slider = st.sidebar.slider('Selecione ponte de corte',0.0, 100.0, (50.0, 50.0))

st.sidebar.title("Train Model")
op1 = st.sidebar.selectbox('Selecionar modelo',('Logistic Regression', 'Decision Tree', 'Random Forrest'))

st.sidebar.write("Select model: ", op1)

df = pd.read_csv("db.csv", delimiter=";", decimal = ",")

lin = df.shape[0]
col = df.shape[1]
st.text("Linhas:")
st.write(lin)
st.text("Colunas:")
st.write(col)
st.write("Conhecendo os dados")
st.write(df.head(10))


if pr:
   st.write("Good\n")
   st.write(df[df.mau==0].describe(percentiles=[.01, .25, .5, .75, .90, .95, .99]).transpose())
   st.write("\nBad\n")
   st.write(df[df.mau==1].describe(percentiles=[.01, .25, .5, .75, .90, .95, .99]).transpose())

if bx:
    numCols = df.select_dtypes(include=[np.number]).columns
    for v in numCols[:-1]:
        #fig, ax = plt.subplots(figsize=(30, 15))
        fig = plt.figure(figsize=[6, 4])
        sns.boxplot(x=v, data=df, orient="v")
        st.pyplot(fig)
        
if cor:
    st.write("Correlação das variáveis")
    corr = df.iloc[:,:-1].corr()
    fig = plt.figure(figsize=[6, 4])
    sns.heatmap(corr,vmax = 0.8, annot = True)
    st.pyplot(fig)


if c1:
    qtd = df['mau'].value_counts()
    prc = df['mau'].value_counts(normalize=True)
    print("Mau:", qtd[1], "Bom:", qtd[0])
    print("Mau: " +"{:.1%}".format(prc[1]), "Bom: " + "{:.1%}".format(prc[0]))
    fig = plt.figure(figsize=[6, 4])
    sns.countplot(y = df['mau']).set_title('#Maus')
    st.pyplot(fig)


execute1 = st.sidebar.button('Executar modelo')


if execute1:
    status_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)

    
    for i in range(100):
        # Update progress bar.
        new_rows = np.random.randn(10, 2)
        status_text.text("Executando modelo")
        progress_bar.progress(i + 1)
    
    
        # Pretend we're doing some computation that takes time.
        time.sleep(0.1)
    
    status_text.text('Finalizado!')
    st.balloons()
   
"""
code = def hello():
st.code(code, language='python')
st.dataframe(df) 
"""