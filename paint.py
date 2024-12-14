import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Step 1: Create Sample Dataset
def create_sample_dataset():
    np.random.seed(42)
    dataset_size = 100
    resin_properties = np.random.uniform(0.5, 1.5, dataset_size)
    pigment_concentration = np.random.uniform(10, 50, dataset_size)
    solvent_properties = np.random.uniform(0.1, 1.0, dataset_size)
    temperature = np.random.uniform(15, 30, dataset_size)
    humidity = np.random.uniform(40, 70, dataset_size)

    viscosity_0_1 = resin_properties * 1.2 + pigment_concentration * 0.02 + solvent_properties * 0.8 + np.random.normal(0, 0.1, dataset_size)
    viscosity_10 = resin_properties * 0.8 + pigment_concentration * 0.01 + solvent_properties * 0.5 + np.random.normal(0, 0.1, dataset_size)
    shear_thinning = viscosity_0_1 / viscosity_10
    drying_time = 50 / solvent_properties + temperature * 0.5 - humidity * 0.2 + np.random.normal(0, 5, dataset_size)

    data = pd.DataFrame({
        "Resin_Properties": resin_properties,
        "Pigment_Concentration": pigment_concentration,
        "Solvent_Properties": solvent_properties,
        "Temperature": temperature,
        "Humidity": humidity,
        "Viscosity_0.1": viscosity_0_1,
        "Viscosity_10": viscosity_10,
        "Shear_Thinning_Index": shear_thinning,
        "Drying_Time": drying_time
    })
    return data

# Step 2: Build Gradient Boosting Regressor Models
def train_model(X, y):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model

# Step 3: Streamlit UI
def main():
    st.title("AI-Powered Rheology Property Prediction")

    # Load Data
    st.header("Dataset")
    data = create_sample_dataset()
    st.dataframe(data.head())

    # Split Data
    features = ["Resin_Properties", "Pigment_Concentration", "Solvent_Properties", "Temperature", "Humidity"]
    targets = ["Viscosity_0.1", "Viscosity_10", "Shear_Thinning_Index", "Drying_Time"]

    X = data[features]
    y = data[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Models for Each Target
    models = {}
    predictions = {}
    metrics = []

    st.header("Model Training and Results")

    for target in targets:
        st.subheader(f"Predicting {target}")
        model = train_model(X_train, y_train[target])
        models[target] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        predictions[target] = y_pred
        
        # Metrics
        mse = mean_squared_error(y_test[target], y_pred)
        r2 = r2_score(y_test[target], y_pred)
        metrics.append((target, mse, r2))
        
        # Display Metrics
        st.write(f"Mean Squared Error: {mse:.3f}")
        st.write(f"R-Squared: {r2:.3f}")

    # Summary Metrics Table
    st.header("Summary Metrics")
    metrics_df = pd.DataFrame(metrics, columns=["Target", "MSE", "R-Squared"])
    st.dataframe(metrics_df)

    # Visualization
    st.header("Prediction Visualization")
    for target in targets:
        st.subheader(f"Actual vs Predicted for {target}")
        st.line_chart(pd.DataFrame({"Actual": y_test[target].values, "Predicted": predictions[target]}))

if __name__ == "__main__":
    main()
