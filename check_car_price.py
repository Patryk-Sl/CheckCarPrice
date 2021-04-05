import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


st.header("""
    Car price!!!
""")


df = pd.read_json("mercedes_data.json")


marka=df['marka_pojazdu'].value_counts().index[:5]
model=df['model_pojazu'].value_counts().index[:10]
generacja=df['generacja'].value_counts().index
rodzaj_paliwa=df['rodzaj_paliwa'].value_counts().index
oferta=df['oferta_od'].value_counts().index


col1, col2 = st.beta_columns(2)

marka_pojazdu = col1.selectbox('marka',marka)
model_pojazu = col1.selectbox('model',model)
generacja_pojazdu = col1.selectbox('generacja',generacja)
rodzaj_paliwa_ = col1.selectbox('rodzaj paliwa',rodzaj_paliwa)
oferta_od = col1.selectbox('oferta',oferta)

przebieg=st.sidebar.slider('przebieg', min_value=1.0, max_value=1000000.0, value=float(5000), step=1.0)
rok_produkcji=st.sidebar.slider('rok produkcji', min_value=1950.0, max_value=2021.0, value=float(2015), step=1.0)
pojemnosc_silnika =st.sidebar.slider('pojemnosc silnika', min_value=500.0, max_value=7000.0, value=float(2000), step=1.0)
moc_silnika=st.sidebar.slider('moc silnika', min_value=20.0, max_value=2000.0, value=float(150), step=1.0)




st.header('Specified Input parameters')
st.write('przebieg',przebieg)

st.write('---')


data={'oferta_od':oferta_od,
      'marka_pojazdu':marka_pojazdu,
      'model_pojazu':model_pojazu,
      "generacja_pojazdu":generacja_pojazdu,
      "rok produkcji":rok_produkcji,
      'przebieg':przebieg,
      'pojemnosc cm3':pojemnosc_silnika,
     'rodzaj_paliwa_':rodzaj_paliwa_,
      'moc silnika': moc_silnika}


X1 = pd.DataFrame([data])

X = df.drop(columns=['cena'], axis=1)


X['oferta_od'] = X['oferta_od'].astype('category')
X['oferta_od'] = X['oferta_od'].cat.codes
X['marka_pojazdu'] = X['marka_pojazdu'].astype('category')
X['marka_pojazdu'] = X['marka_pojazdu'].cat.codes
X['model_pojazu'] = X['model_pojazu'].astype('category')
X['model_pojazu'] = X['model_pojazu'].cat.codes
X['generacja'] = X['generacja'].astype('category')
X['generacja'] = X['generacja'].cat.codes
X['rodzaj_paliwa'] = X['rodzaj_paliwa'].astype('category')
X['rodzaj_paliwa'] = X['rodzaj_paliwa'].cat.codes

X1['oferta_od'] = X1['oferta_od'].astype('category')
X1['oferta_od'] = X1['oferta_od'].cat.codes
X1['marka_pojazdu'] = X1['marka_pojazdu'].astype('category')
X1['marka_pojazdu'] = X1['marka_pojazdu'].cat.codes
X1['model_pojazu'] = X1['model_pojazu'].astype('category')
X1['model_pojazu'] = X1['model_pojazu'].cat.codes
X1['generacja_pojazdu'] = X1['generacja_pojazdu'].astype('category')
X1['generacja_pojazdu'] = X1['generacja_pojazdu'].cat.codes
X1['rodzaj_paliwa_'] = X1['rodzaj_paliwa_'].astype('category')
X1['rodzaj_paliwa_'] = X1['rodzaj_paliwa_'].cat.codes



Y=df['cena']

model = RandomForestRegressor()
model.fit(X, Y)

prediction = model.predict(X1)
st.header('Prediction of price')
st.write(prediction)