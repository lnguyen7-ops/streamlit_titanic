import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import streamlit as st
import joblib

st.title("Our last morning kick off :sob:")

# data science stuff
# load titanic data
@st.cache
def load_data(path):
    return pd.read_csv(path)
df = load_data("titanic_dataset.csv")

# make some plot
survived = df["survived"].value_counts(normalize=True)
# using quick streamlit way
#st.bar_chart(survived)
# using matplotlib
fig, ax = plt.subplots()
ax.bar(survived.index, survived.values)
ax.set_xticks([0,1])
ax.set_xticklabels(["No", "Yes"])
ax.set_xlabel("Survived")
ax.set_ylabel("Ratio")
ax.set_title("Titanic Historical Data")
st.pyplot(fig)

# import model for prediction
scaler_mm = joblib.load("scaler_mm_titanic")
logreg = joblib.load("logreg_titanic")
'# Passenger information'
# inputs
# age
age = st.number_input(label="Age", min_value=0, step=1)
# embark
embark_opts = ["Cherbourg", "Queenstown", "Southampton"]
embark_select = st.selectbox(label="Port of Embarkation", options=embark_opts)
embark_onehot = [1 if embark_select==option else 0 for option in embark_opts]
# male?
male = st.checkbox("Male?")
male_int = int(male)
# travel alone
travel_alone = st.checkbox("Travel alone?")
travel_alone_int = int(travel_alone)
# ticket price
price_opts = ["0 - 50", "51 - 100", "100 and above"]
price_select = st.selectbox(label="Ticket price ($)", options=price_opts)
price_group = price_opts.index(price_select) + 1
#model input
passenger_info = [age] + embark_onehot[1:] + [male_int, travel_alone_int, price_group]

#scale
x = scaler_mm.transform(np.array([passenger_info]))
pred = logreg.predict_proba(x)
'''# Probability'''
f"## Survived :smile:: {pred[0,1]*100:2.4}"
f"## Not survived :worried:: {pred[0,0]*100:2.4}"
