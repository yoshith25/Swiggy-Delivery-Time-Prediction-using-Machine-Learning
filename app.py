
import streamlit as st
import pandas as pd
import pickle

st.image(r"C:\Users\vyosh\Desktop\innomatic\Swiggy Delvery Time Prediction ML project\swiggy_banner.png.png", use_container_width=True)

# Title
st.title("🛵 Swiggy Delivery Time Prediction Using Machine Learning")
st.write("Enter the order details below to predict the estimated delivery time (ETA).")

# Load model pipeline
@st.cache_resource
def load_model():
    with open('Swiggy_delivery_time_required.pkl', 'rb') as f:
        return pickle.load(f)

pipeline = load_model()

# User Inputs
st.subheader("Order & Delivery Details")
col1, col2, col3 = st.columns(3)

with col1:
    weather = st.selectbox("Weather Condition", ['sunny', 'cloudy', 'fog', 'stormy', 'sandstorms', 'windy'])
    age = st.slider("Rider Age", min_value=18.0, max_value=65.0, value=28.0, step=1.0)
    ratings = st.slider("Rider Rating (1.0 - 5.0)", min_value=1.0, max_value=5.0, value=4.7, step=0.1)
    traffic = st.selectbox("Traffic Level", ['low', 'medium', 'high', 'jam'])
    vehicle_condition = st.selectbox("Vehicle Condition", [0, 1, 2, 3])
    type_of_order = st.selectbox("Type of Order", ['snack', 'meal', 'drinks', 'buffet'])

with col2:
    type_of_vehicle = st.selectbox("Type of Vehicle", ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
    multiple_deliveries = st.slider("Multiple Deliveries Assigned", min_value=0.0, max_value=3.0, value=1.0, step=1.0)
    festival = st.selectbox("Festival Day", ['no', 'yes'])
    city_type = st.selectbox("City Type", ['semi-urban', 'urban', 'metropolitian'])
    city_name = st.selectbox("City Name / Code", ['BANG', 'HYD', 'CHEN', 'MUM', 'DEL', 'COIMB', 'PUNE', 'KOL', 'IND', 'SUR', 'AGR', 'JAP', 'LUDH', 'AURG'])
    order_day = st.slider("Order Day of Month", min_value=1, max_value=31, value=15, step=1)

with col3:
    order_day_of_week = st.selectbox("Day of Week", ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'])
    order_month = st.slider("Order Month", min_value=1, max_value=12, value=3, step=1)
    pickup_time_minutes = st.slider("Kitchen Preparation / Pickup Time (mins)", min_value=1.0, max_value=60.0, value=12.0, step=1.0)
    order_time_hour = st.slider("Order Hour (0 - 23)", min_value=0.0, max_value=23.0, value=20.0, step=1.0)
    distance = st.slider("Delivery Distance (km)", min_value=0.1, max_value=50.0, value=8.5, step=0.5)

# Derived fields
is_weekend = 1 if order_day_of_week in ['saturday', 'sunday'] else 0

if 5 <= order_time_hour < 12:
    order_time_of_day = 'morning'
elif 12 <= order_time_hour < 17:
    order_time_of_day = 'afternoon'
elif 17 <= order_time_hour < 21:
    order_time_of_day = 'evening'
elif 21 <= order_time_hour <= 23:
    order_time_of_day = 'night'
else:
    order_time_of_day = 'after_midnight'

# Prediction Button & Output Display
if st.button("Predict Delivery Time"):
    input_data = pd.DataFrame([{
        'age': age,
        'ratings': ratings,
        'weather': weather,
        'traffic': traffic,
        'vehicle_condition': vehicle_condition,
        'type_of_order': type_of_order,
        'type_of_vehicle': type_of_vehicle,
        'multiple_deliveries': multiple_deliveries,
        'festival': festival,
        'city_type': city_type,
        'city_name': city_name,
        'order_day': order_day,
        'order_month': order_month,
        'order_day_of_week': order_day_of_week,
        'is_weekend': is_weekend,
        'pickup_time_minutes': pickup_time_minutes,
        'order_time_hour': order_time_hour,
        'order_time_of_day': order_time_of_day,
        'distance': distance
    }])

    prediction = pipeline.predict(input_data)[0]

    st.subheader("🛵 Delivery Prediction Results")

    st.metric("📍 Delivery Distance", f"{distance:.2f} km")

    st.metric("⏱️ Estimated Delivery Time", f"{prediction:.2f} minutes")

    st.success(
        f"Expected Delivery Window: "
        f"{max(5.0, prediction - 3.5):.0f} – "
        f"{prediction + 3.5:.0f} minutes"
    )

    st.write(f"**Rider Rating:** {ratings:.1f} / 5.0")
