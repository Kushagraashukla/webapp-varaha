# --- Re-declare everything your pipeline needs ---
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import dill

# same as training
low_impact = ['How Long TV PC Daily Hour', 'How Long Internet Daily Hour']
multi_cols = ['Recycling', 'Cooking_With']
onehot_cols = ['Body Type','Diet','Heating Energy Source','How Often Shower',
               'Transport','Vehicle Type','Social Activity','Waste Bag Size',
               'Frequency of Traveling by Air','Energy efficiency']

def drop_cols(df):
    return df.drop(columns=low_impact)

class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.encoders = {}
    def fit(self, X, y=None):
        from sklearn.preprocessing import MultiLabelBinarizer
        for c in self.cols:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[c].apply(lambda x: x.split(',') if isinstance(x, str) else []))
            self.encoders[c] = mlb
        return self
    def transform(self, X):
        X = X.copy()
        for c, mlb in self.encoders.items():
            arr = mlb.transform(X[c].apply(lambda x: x.split(',') if isinstance(x, str) else []))
            new_cols = [f"{c}_{cls}" for cls in mlb.classes_]
            X = pd.concat([X.drop(columns=[c]),
                           pd.DataFrame(arr, columns=new_cols, index=X.index)],
                          axis=1)
        return X

import streamlit as st
with open("C:/Users/OMEN/OneDrive/Documents/machine learning/ML projects/carbonfootprintpredictor/trained_model.sav", "rb") as f:
    loaded_model = dill.load(f)

st.title("Carbon Footprint Predictor")

# ---------- User inputs ----------
sex          = st.selectbox("Sex", ["male", "female"])
body_type    = st.selectbox("Body Type", ["obese","overweight","underweight","normal"])
diet         = st.selectbox("Diet", ["omnivore","vegetarian","vegan","pescatarian"])
shower       = st.selectbox("How Often Shower",
                            ["daily","less frequently","more frequently","twice a day"])
heating      = st.selectbox("Heating Energy Source",
                            ["electricity","natural gas","wood","coal"])
transport    = st.selectbox("Transport", ["public","private","walk/bicycle"])
vehicle_type = st.selectbox("Vehicle Type",
                            ["petrol","diesel","electric","hybrid","lpg","unknown"])
social       = st.selectbox("Social Activity", ["never","sometimes","often"])
grocery      = st.number_input("Monthly Grocery Bill", min_value=0)
air_freq     = st.selectbox("Frequency of Traveling by Air",
                            ["never","rarely","frequently","very frequently"])
vehicle_km   = st.number_input("Vehicle Monthly Distance Km", min_value=0)
bag_size     = st.selectbox("Waste Bag Size", ["small","medium","large","extra large"])
bag_count    = st.number_input("Waste Bag Weekly Count", min_value=0)
tv_hour      = st.number_input("How Long TV PC Daily Hour", min_value=0)
new_clothes  = st.number_input("How Many New Clothes Monthly", min_value=0)
internet_hr  = st.number_input("How Long Internet Daily Hour", min_value=0)
energy_eff   = st.selectbox("Energy efficiency", ["Yes","No","Sometimes"])
recycling    = st.multiselect("Recycling", ["Paper","Plastic","Glass","Metal"])
cooking      = st.multiselect("Cooking With",
                              ["Stove","Oven","Microwave","Grill","Airfryer"])

# ---------- Prediction ----------
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Sex": sex,
        "Body Type": body_type,
        "Diet": diet,
        "How Often Shower": shower,
        "Heating Energy Source": heating,
        "Transport": transport,
        "Vehicle Type": vehicle_type,
        "Social Activity": social,
        "Monthly Grocery Bill": grocery,
        "Frequency of Traveling by Air": air_freq,
        "Vehicle Monthly Distance Km": vehicle_km,
        "Waste Bag Size": bag_size,
        "Waste Bag Weekly Count": bag_count,
        "How Long TV PC Daily Hour": tv_hour,
        "How Many New Clothes Monthly": new_clothes,
        "How Long Internet Daily Hour": internet_hr,
        "Energy efficiency": energy_eff,
        "Recycling": recycling,       # keep as list for MultiLabelEncoder
        "Cooking_With": cooking       # keep as list
    }])

    pred = loaded_model.predict(input_df)[0]
    st.success(f"Estimated Carbon Footprint: {pred:.2f}")
