import numpy as np
import pandas as pd
import pickle
import streamlit as st

with open(r"C:/Users/OMEN/OneDrive/Documents/machine learning/ML projects/carbonfootprintpredictor/trained_model.sav","rb") as f:
    loaded_model = pickle.load(f)

def pre(input_df):
    try:
        return loaded_model.predict(input_df)[0]
    except:
        return None

feature_names = [
    "Sex","Monthly Grocery Bill","Vehicle Monthly Distance Km","Waste Bag Weekly Count","How Many New Clothes Monthly",
    "Recycling Glass","Recycling Metal","Recycling Paper","Recycling Plastic",
    "Cooking with Airfryer","Cooking with Grill","Cooking with Microwave","Cooking with Oven","Cooking with Stove",
    "Body Type Obese","Body Type Overweight","Body Type Underweight",
    "Diet Pescatarian","Diet Vegan","Diet Vegetarian",
    "Heating Energy Source Electricity","Heating Energy Source Natural Gas","Heating Energy Source Wood",
    "How Often Shower Less Frequently","How Often Shower More Frequently","How Often Shower Twice a Day",
    "Transport Public","Transport Walk/Bicycle",
    "Vehicle Type Electric","Vehicle Type Hybrid","Vehicle Type LPG","Vehicle Type Petrol","Vehicle Type Unknown",
    "Social Activity Often","Social Activity Sometimes",
    "Waste Bag Size Large","Waste Bag Size Medium","Waste Bag Size Small",
    "Frequency of Traveling by Air Never","Frequency of Traveling by Air Rarely","Frequency of Traveling by Air Very Frequently",
    "Energy Efficiency Sometimes","Energy Efficiency Yes"
]

def main():
    st.title("Varaha Web App")
    sex=st.selectbox("Sex (0=Female,1=Male)",[0,1])
    grocery=st.number_input("Monthly Grocery Bill",min_value=0)
    vehicle_km=st.number_input("Vehicle Monthly Distance Km",min_value=0)
    waste_bag=st.number_input("Waste Bag Weekly Count",min_value=0)
    clothes=st.number_input("How Many New Clothes Monthly",min_value=0)
    rec_glass=st.selectbox("Recycling Glass",[False,True])
    rec_metal=st.selectbox("Recycling Metal",[False,True])
    rec_paper=st.selectbox("Recycling Paper",[False,True])
    rec_plastic=st.selectbox("Recycling Plastic",[False,True])
    cook_airfryer=st.selectbox("Cooking with Airfryer",[False,True])
    cook_grill=st.selectbox("Cooking with Grill",[False,True])
    cook_microwave=st.selectbox("Cooking with Microwave",[False,True])
    cook_oven=st.selectbox("Cooking with Oven",[False,True])
    cook_stove=st.selectbox("Cooking with Stove",[False,True])
    body_obese=st.selectbox("Body Type Obese",[False,True])
    body_overweight=st.selectbox("Body Type Overweight",[False,True])
    body_underweight=st.selectbox("Body Type Underweight",[False,True])
    diet_pesc=st.selectbox("Diet Pescatarian",[False,True])
    diet_vegan=st.selectbox("Diet Vegan",[False,True])
    diet_veg=st.selectbox("Diet Vegetarian",[False,True])
    heat_electric=st.selectbox("Heating Energy Source Electricity",[False,True])
    heat_gas=st.selectbox("Heating Energy Source Natural Gas",[False,True])
    heat_wood=st.selectbox("Heating Energy Source Wood",[False,True])
    shower_less=st.selectbox("How Often Shower Less Frequently",[False,True])
    shower_more=st.selectbox("How Often Shower More Frequently",[False,True])
    shower_twice=st.selectbox("How Often Shower Twice a Day",[False,True])
    transport_public=st.selectbox("Transport Public",[False,True])
    transport_walk=st.selectbox("Transport Walk/Bicycle",[False,True])
    vehicle_electric=st.selectbox("Vehicle Type Electric",[False,True])
    vehicle_hybrid=st.selectbox("Vehicle Type Hybrid",[False,True])
    vehicle_lpg=st.selectbox("Vehicle Type LPG",[False,True])
    vehicle_petrol=st.selectbox("Vehicle Type Petrol",[False,True])
    vehicle_unknown=st.selectbox("Vehicle Type Unknown",[False,True])
    social_often=st.selectbox("Social Activity Often",[False,True])
    social_sometimes=st.selectbox("Social Activity Sometimes",[False,True])
    bag_large=st.selectbox("Waste Bag Size Large",[False,True])
    bag_medium=st.selectbox("Waste Bag Size Medium",[False,True])
    bag_small=st.selectbox("Waste Bag Size Small",[False,True])
    air_never=st.selectbox("Frequency of Traveling by Air Never",[False,True])
    air_rarely=st.selectbox("Frequency of Traveling by Air Rarely",[False,True])
    air_veryfreq=st.selectbox("Frequency of Traveling by Air Very Frequently",[False,True])
    energy_sometimes=st.selectbox("Energy Efficiency Sometimes",[False,True])
    energy_yes=st.selectbox("Energy Efficiency Yes",[False,True])

    if st.button("Predict"):
        input_list=[sex,grocery,vehicle_km,waste_bag,clothes,
            rec_glass,rec_metal,rec_paper,rec_plastic,
            cook_airfryer,cook_grill,cook_microwave,cook_oven,cook_stove,
            body_obese,body_overweight,body_underweight,
            diet_pesc,diet_vegan,diet_veg,
            heat_electric,heat_gas,heat_wood,
            shower_less,shower_more,shower_twice,
            transport_public,transport_walk,
            vehicle_electric,vehicle_hybrid,vehicle_lpg,vehicle_petrol,vehicle_unknown,
            social_often,social_sometimes,
            bag_large,bag_medium,bag_small,
            air_never,air_rarely,air_veryfreq,
            energy_sometimes,energy_yes]
        df_input=pd.DataFrame([input_list],columns=feature_names)
        pred=pre(df_input)
        if pred is None:
            st.error("Invalid input! Please check your entries.")
        else:
            st.success(f"The average carbon footprint of the person is {pred}")

if __name__=='__main__':
    main()
