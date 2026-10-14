import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cleaned car.xls")
MODEL_PATH = os.path.join(BASE_DIR, "LinearRegression.pkl")


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# Load trained model
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


# Application title
st.title("🚗 Car Price Prediction")
st.write(
    "Enter the details of your car to predict its estimated price."
)

st.divider()

# Sidebar inputs
st.sidebar.header("Enter Car Details")

company = st.sidebar.selectbox(
    "Select Company",
    sorted(df["company"].dropna().unique())
)

car_names = sorted(
    df.loc[df["company"] == company, "name"].dropna().unique()
)

name = st.sidebar.selectbox(
    "Select Car Model",
    car_names
)

year = st.sidebar.selectbox(
    "Manufacturing Year",
    sorted(
        df["year"].dropna().astype(int).unique(),
        reverse=True
    )
)

kms_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=1000000,
    value=30000,
    step=1000
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    sorted(df["fuel_type"].dropna().unique())
)

# Prediction button
if st.sidebar.button("Predict Price", type="primary"):

    input_data = pd.DataFrame([{
        "name": name,
        "company": company,
        "year": year,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type
    }])

    try:
        # Predict using the trained model
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        st.success(
            f"Estimated Car Price: ₹ {prediction:,.0f}"
        )

        st.write("### Car Details")
        st.dataframe(input_data, use_container_width=True)

        st.caption(
            "This price is an estimate from your machine learning model."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

        st.info(
            "Your model's input features and preprocessing must "
            "match the features used during training."
        )


# Dataset preview
with st.expander("View Car Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

# Footer
st.divider()
st.caption("Car Price Prediction | Linear Regression")