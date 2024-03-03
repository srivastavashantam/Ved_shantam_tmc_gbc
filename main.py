import streamlit as st
import pandas as pd
import plotly.express as px
import pickle as pickle


# Load the pre-trained model and standardization scaler
model = pickle.load(open("lightgbm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
onehot_encoder = pickle.load(open("onehot_encoder.pkl", "rb"))

# Define numerical and categorical features
numerical_features = ['AGE_random', 'BMI_DETAILS_random', 'BILIRUBIN_CONJUGATED_random',
                      'BILIRUBIN_UNCONJUGATED_random', 'BILIRUBIN_TOTAL_random', 'AST_SGOT_random',
                      'ALT_SGPT_random', 'ALK_PHOS_ALP_random', 'GAMMA_GT_random', 'ALBUMIN_random']

categorical_features = ['Thyroidism', 'HTN', 'Family_cancer_history', 'ECOG_final', 'Gall_stone_final',
                        'Pathology', 'Tumour_grade_final', 'Cancer_presentation_final', 'Tumour_stage_final',
                        'Mets_site_final', 'Surgery_final', 'Treatment_final', 'GENDER', 'Diabetes_final']

def get_clean_data():
    data = pd.read_excel("sem2_working_gbc_clean_final.xlsx")
    return data

def plot_numerical_feature_variation(data, feature):
    fig = px.histogram(data, x=feature, color='Status', marginal="box", barmode="overlay")
    fig.update_layout(title=f'Variation of "{feature}" with "Status"', xaxis_title=feature, yaxis_title='Count')
    return fig

def plot_categorical_feature_variation(data, feature):
    fig = px.histogram(data, x=feature, color='Status', barmode='group')
    fig.update_layout(title=f'Variation of "{feature}" with "Status"', xaxis_title=feature, yaxis_title='Count')
    return fig

def predict_survival(input_data):

    input_data = pd.DataFrame([input_data])

    numerical_cols = [
        'AGE_random', 'BMI_DETAILS_random', 'BILIRUBIN_CONJUGATED_random', 'BILIRUBIN_UNCONJUGATED_random', 'BILIRUBIN_TOTAL_random',
        'AST_SGOT_random', 'ALT_SGPT_random', 'ALK_PHOS_ALP_random', 'GAMMA_GT_random', 'ALBUMIN_random'
    ]
    
    # Scale the input data
    scaled_data = scaler.transform(input_data[numerical_cols])

    categorical_columns = [
        'Thyroidism', 'HTN', 'Family_cancer_history', 'ECOG_final', 'Gall_stone_final', 'Pathology', 'Tumour_grade_final',
        'Cancer_presentation_final', 'Tumour_stage_final', 'Mets_site_final', 'Surgery_final', 'Treatment_final','GENDER',
        'Diabetes_final'
    ]

    # Onehot encode
    encoded_data = onehot_encoder.transform(input_data[categorical_columns])

    final_data = pd.concat([
        pd.DataFrame(scaled_data), 
        pd.DataFrame(encoded_data)
    ], axis=1
    )

    # Predict using the model
    prediction = model.predict_proba(final_data)[0]
    return prediction

def visualization_page(data):
    st.header("Exploratory Data Analysis")
    st.sidebar.title("EDA of Features")
    feature_type = st.sidebar.selectbox("Select Feature Type", ["Numerical", "Categorical"])

    if feature_type == "Numerical":
        st.sidebar.header("Numerical Features")
        selected_feature = st.sidebar.selectbox("Select Numerical Feature", numerical_features)
        st.plotly_chart(plot_numerical_feature_variation(data, selected_feature))

    elif feature_type == "Categorical":
        st.sidebar.header("Categorical Features")
        selected_feature = st.sidebar.selectbox("Select Categorical Feature", categorical_features)
        st.plotly_chart(plot_categorical_feature_variation(data, selected_feature))



def prediction_page(data):
    st.header("Predict Survival")
    with st.form("prediction_form"):
        numerical_inputs = {}
        categorical_inputs = {}

        # Input fields for numerical features
        st.subheader("Numerical Features")
        for feature in numerical_features:
            numerical_inputs[feature] = st.number_input(f"Enter {feature}", min_value=0)

        # Input fields for categorical features
        st.subheader("Categorical Features")
        for feature in categorical_features:
            categorical_inputs[feature] = st.selectbox(f"Select {feature}", data[feature].unique())

        submitted = st.form_submit_button("Predict")

        if submitted:

            input_data = {**numerical_inputs, **categorical_inputs}
            prediction = predict_survival(input_data)
            st.write(f"The patient's predicted survival probability:")
            st.write(f"- Dead: {prediction[0]:.2f}")
            st.write(f"- Alive: {prediction[1]:.2f}")


            # data = data[data['Status'] != 'Dead'].reset_index(drop=True)
            # for i in range(len(data)):  
            #     print(i, '\n')
            #     prediction = predict_survival(data.iloc[i])
            #     st.write(f"The patient's predicted survival probability:")
            #     st.write(f"- Status: {data.iloc[i]['Status']}")
            #     st.write(f"- Dead: {prediction[0]:.2f}")
            #     st.write(f"- Alive: {prediction[1]:.2f}")

def main():
    st.set_page_config(
        page_title='Gallbladder Cancer Predictor',
        page_icon=':female-doctor:',
        layout='wide'
    )

    data = get_clean_data()

    st.title("Gallbladder Cancer Predictor")
    st.write("Welcome to our webapp")

    # Sidebar for choosing page
    page = st.sidebar.radio("Navigation", options=["Visualization", "Prediction"])

    # Display selected page
    if page == "Visualization":
        visualization_page(data)
    elif page == "Prediction":
        prediction_page(data)

if __name__ == '__main__':
    main()
