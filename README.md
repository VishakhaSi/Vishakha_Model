# Vishakha_Model
This repository contains the implementation of a machine learning model to predict rheological properties of coatings, such as viscosity, shear-thinning behavior, and drying time, using historical formulation data. The project leverages supervised learning algorithms like Gradient Boosting Machines.
# Predicting Rheological Properties of Coatings using AI/ML  

This repository hosts the code and resources for a project focused on predicting key rheological properties of coatings using supervised machine learning models. The project aims to provide a data-driven approach for optimizing paint formulations, reducing trial-and-error experiments, and accelerating innovation in the paints and coatings industry.

## **Project Overview**  
Rheological properties such as viscosity, shear-thinning behavior, and drying time are critical for paint performance, storage stability, and customer satisfaction. Traditional methods for predicting these properties are time-consuming and resource-intensive. This project addresses these challenges by proposing a Gradient Boosting Machine (GBM)-based solution to predict these properties based on formulation inputs such as resin properties, pigment concentrations, and solvent compositions.

### **Key Features**  
- **Data Preprocessing**: Handles missing data, normalizes variables, and creates engineered features like pigment-to-binder ratios and environmental factor interactions.  
- **Supervised ML Model**: Implements Gradient Boosting Machines, optimized for predicting viscosity and drying time.  
- **Model Validation**: Includes cross-validation and evaluation metrics such as Mean Squared Error (MSE) and R-squared (R²).  
- **Scalable Approach**: Designed to integrate seamlessly with historical formulation data and expand for future applications.  

## **Project Structure**  
- **paint.py**: The python code for model.   
- **`README.md`**: Detailed project documentation.  

## **Getting Started**  

### **Prerequisites**  
To run this project,:
- sudo apt update
- sudo apt install python3 python3-pip -y
- pip3 install numpy pandas scikit-learn streamlit
- streamlit run paint.py


