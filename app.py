import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from PIL import Image

classifier = pickle.load(open("LRMODEL.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

#st.title("Titanic Prediction")

def prediction(Personal_Fare, Pclass_1, Pclass_2, Pclass_3, Sex_female,
       Sex_male, Embarked_C, Embarked_Q, Embarked_S,
       Family_Type_alone, Family_Type_large, Family_Type_small,
       Age_Type_Child, Age_Type_Mid_aged,Age_Type_Missing, Age_Type_Old, Age_Type_Young,
       Deck_A, Deck_B, Deck_C, Deck_D, Deck_E, Deck_F, Deck_G,
       Deck_M, Deck_T):
    scaled_data = scaler.transform(
        [[Personal_Fare, Pclass_1, Pclass_2, Pclass_3, Sex_female,
       Sex_male, Embarked_C, Embarked_Q, Embarked_S,
       Family_Type_alone, Family_Type_large, Family_Type_small,
       Age_Type_Child, Age_Type_Mid_aged,Age_Type_Missing, Age_Type_Old, Age_Type_Young,
       Deck_A, Deck_B, Deck_C, Deck_D, Deck_E, Deck_F, Deck_G,
       Deck_M, Deck_T]])
    prediction = classifier.predict(scaled_data)
    print(prediction)
    return prediction

#def main():
    # giving the webpage a title
st.title("Titanic Death Prediction")

html_temp = """ 
<div style ="background-color:yellow;padding:13px"> 
<h1 style ="color:black;text-align:center;">Titanic Death or Survival Prediction</h1> 
</div> 
"""

# this line allows us to display the front end aspects we have
# defined in the above code
st.markdown(html_temp, unsafe_allow_html=True)




Personal_Fare = st.number_input("Personl Fare",)
Pclass_1 = st.number_input("Pclass 1",)
Pclass_2 = st.number_input("Pclass 2",)
Pclass_3 = st.number_input("Pclass 3",)
Sex_female = st.number_input("Male",)
Sex_male = st.number_input("Female",)
Embarked_C = st.number_input("Cherbourg Station",)
Embarked_Q = st.number_input("Queenstown Station",)
Embarked_S = st.number_input("Southampton Station",)
Family_Type_alone = st.number_input("Travelling Alone",)
Family_Type_large = st.number_input("Travelling with more than 4 ",)
Family_Type_small = st.number_input("Travelling with more than 2 and less than 5 ",)
Age_Type_Child = st.number_input("Child",)
Age_Type_Mid_aged = st.number_input("Middle Ages",)
Age_Type_Missing = st.number_input("Missing Age",)
Age_Type_Old = st.number_input("Old aged",)
Age_Type_Young = st.number_input("Young aged",)
Deck_A = st.number_input("Deck_A",)
Deck_B = st.number_input("Deck_B",)
Deck_C = st.number_input("Deck_C",)
Deck_D = st.number_input("Deck_D",)
Deck_E = st.number_input("Deck_E",)
Deck_F = st.number_input("Deck_F",)
Deck_G = st.number_input("Deck_G",)
Deck_M = st.number_input("Deck_M",)
Deck_T = st.number_input("Deck_T",)
result = ""


if st.button("Predict"):
    result = prediction(Personal_Fare, Pclass_1, Pclass_2, Pclass_3, Sex_female,
    Sex_male, Embarked_C, Embarked_Q, Embarked_S,
    Family_Type_alone, Family_Type_large, Family_Type_small,
    Age_Type_Child, Age_Type_Mid_aged,Age_Type_Missing, Age_Type_Old, Age_Type_Young,
    Deck_A, Deck_B, Deck_C, Deck_D, Deck_E, Deck_F, Deck_G,
    Deck_M, Deck_T)
if result == 1.0:
    st.success('The Passenger Survived')
else :
    st.success('The Passenger Died')

    #if __name__ == '__main__':
       # main()

