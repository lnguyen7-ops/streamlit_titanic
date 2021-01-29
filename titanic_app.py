import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import streamlit as st
import joblib

# Create a title
st.title("Our last morning kick off :sob:")
# You can also use markdown syntax. Un-comment the line below to test it out
#st.write('# Out last morning kick off :sob:')

# Now let's do some data science stuff
# Create a function to load the titanic data
@st.cache # use cache to store data for reuse, hence helps app run faster.
def load_data(path):
    return pd.read_csv(path)
# Load the titanic dataset
df = load_data("titanic_dataset.csv")

# Let's make a plot
# Ratio of survived and not survived passenger
survived = df["survived"].value_counts(normalize=True)
# Make bar_chart using quick streamlit way
#st.bar_chart(survived)
# Make bar chart using matplotlib
fig, ax = plt.subplots(figsize=(3,3)) # create the figure object with one axis
ax.bar(survived.index, survived.values) # make bar chart
ax.set_xticks([0,1]) # set tick mart at value 0 and 1
ax.set_xticklabels(["No", "Yes"]) # change tick mark label to "No" and "Yes"
ax.set_xlabel("Survived")
ax.set_ylabel("Ratio")
ax.set_title("Titanic Historical Data")
st.pyplot(fig) # Tell streamlit to render this pyplot object

# import model for prediction
@st.cache # store model for reuse
def load_scaler_and_model(scaler, model):
    return joblib.load(scaler), joblib.load(model)
scaler_mm, logreg = load_scaler_and_model("scaler_mm_titanic", "logreg_titanic")

'# Passenger information'
# User inputs
# Age of passenger
age = st.number_input(label="Age", min_value=0, step=1)
# Point of embark
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
# model input
# This model we loaded requires the following input format.
# Input is an array of passenger information 
# [age, Queenstown, Southhampton, male, travel_alone, price_group]
passenger_info = [age] + embark_onehot[1:] + [male_int, travel_alone_int, price_group]

# Scale
x = scaler_mm.transform(np.array([passenger_info]))
# Predict
pred = logreg.predict_proba(x)
# Show output
'# Probability'
f"## Survived :smile:: {pred[0,1]*100:2.4}%"
f"## Not survived :worried:: {pred[0,0]*100:2.4}%"
