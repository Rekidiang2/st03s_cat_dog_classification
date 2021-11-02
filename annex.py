import streamlit as st
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import cv2 as cv
import pickle
import sqlite3
#import matplotlib as plt

# == Logo
def logo():
    # source: jason-leung from unsplash
    logo = "images/rkd_logo_fin.png"
    image = Image.open(logo)
    img_logo = np.array(image) 
    img_shape = (150, 100)
    img_logo = cv.resize(img_logo, img_shape, interpolation=cv.INTER_AREA)
    st.sidebar.image(img_logo)
logo()
# == Home =======================================================================================
def home():
    st.markdown("""
   **Diabetes** is one of the diseases that affects many people in the world, detecting it early will allow effective 
   care taking of patient. This application  allows automatic and rapid prediction of diabetes in **prediabetic stage** 
   using certain symptom measurements.
    """)

    image = Image.open('images/diabetes_img.png')
    image = np.array(image) 
    img_shape = (120, 30)
    image = cv.resize(image, img_shape, interpolation=cv.INTER_AREA)
    st.image(image, caption='Machine Learning Project by Rekidiang Data', use_column_width=True)

    st.markdown("""
        To navigate the application, in slider bar  select **About** to have info about the project, **App** to detect if patient have 
        diabetes or not accordingly to symptom measurement, **Analysis** to know more about data, analysis and ML model
        use in this project and **Prediction Result** to see results records.

        """)


# == App ======================================================================================
def app():
    # create data base
    # conn = sqlite3.connect(':memory:')  #temporary data base
    conn = sqlite3.connect('diabetes_result.db')

    # create table

    c = conn.cursor()  # create a cursor
    # data type : NULL, INTEGER, REAL, TEXT, BLOB

    '''
    c.execute("""CREATE TABLE patients(
    PatientID text,
    Name text, 
    Age text, 
    Gender text,
    Pregnancies integer, 
    Glucose real,
    Insulin real,
    BMI real,
    BP real,
    DPF real,
    SkinTickness integer,
    Result text

    )""")

    '''
    # --------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        patientID = st.text_input("Patient ID")
    with col2:
        name = st.text_input("Patient Name")
    with col3:
        age = st.text_input("Age")
    with col4:
        # user_input = st.text_input("Patient Name")
        gender = st.radio('Gender', ['Female', 'Male'])

    col5, col6 = st.columns(2)
    with col5:
        preg = st.slider('Number of Pregnancies', 0, 20, 0)
    with col6:
        insu = st.slider('Insulin', 0.0, 700.0, 220.0)

    col7, col8 = st.columns(2)
    with col7:
        gluco = st.slider('Glucose', 0, 200, 110)
    with col8:
        bmi = st.slider('BMI', 0.0, 60.0, 30.0)
    col9, col10 = st.columns(2)
    with col9:
        dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.0, 0.08)
    with col10:
        skin = st.slider('Skin Thickness', 1, 95, 1)
    col9, col10 = st.columns(2)
    with col9:
        bp = st.slider('Blood Pressure', 0, 100, 70)

    # DATA
    # st.write(patientID, name, age, gender, preg1, insu, gluco, bmi, dpf, skin, bp)
    data1 = {"preg": preg, "gluco": gluco, "bp": bp, "skin": skin, "insu": insu, "bmi": bmi, "dpf": dpf, "age": age}
    data = [preg, gluco, bp, skin, insu, bmi, dpf, age]

    data_dict = {"patientID": patientID, "name": name, "gender": gender, "preg": preg, "gluco": gluco, "bp": bp,
                 "skin": skin, "insu": insu, "bmi": bmi, "dpf": dpf, "age": age}

    # FOR DATABASE
    patientID = str(data_dict['patientID'])
    name = data_dict['name']
    age = data_dict['age']
    gender = data_dict['gender']
    preg = data_dict['preg']
    gluco = data_dict['gluco']
    bp = data_dict['bp']
    skin = data_dict['skin']
    insu = data_dict['insu']
    bmi = data_dict['bmi']
    dpf = data_dict['dpf']

    st.markdown("---")

    if st.button("Detect"):
        model_path = "models/diabdetect_lda_83.sav"
        model = pickle.load(open(model_path, "rb"))
        pred_result = model.predict([data])

        if pred_result[0] == 0:
            st.success("Result is negative")
            data_dict["result"] = "Negative"
            result = data_dict['result']
            # st.write(data_dict['patientID'], data_dict['name'], data_dict['gender'], data_dict['preg'], data_dict['gluco'], data_dict['bp'],data_dict['skin'], data_dict['insu'], data_dict['bmi'], data_dict['dpf'], data_dict['age'], data_dict['Result'])
            df = pd.DataFrame(data_dict, index=[0])

            param = (patientID, name, age, gender, preg, gluco, bp, skin, insu, bmi, dpf, result)
            c.execute("INSERT INTO patients VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", param)
            st.write("Result Details : ")
            st.table(df)
            # insert data

        else:
            st.success("Result is Positive")
            data_dict["result"] = "Positive"
            result = data_dict["result"]
            df = pd.DataFrame(data_dict, index=[0])

            param = (patientID, name, age, gender, preg, gluco, bp, skin, insu, bmi, dpf, result)
            c.execute("INSERT INTO patients VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", param)
            st.write("Result Details : ")
            st.table(df)

        # commit our command
        conn.commit()
        # close connection
        conn.close()


# == Analysis =============================================================================================
def analysis():
    st.markdown("""
    ## 1. Data

    This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is 
    to predict based on diagnostic measurements whether a patient has diabetes.

    * Number of Instances: 768
    * Number of Attributes: 8 plus class
    * For Each Attribute: (all numeric-valued)
    * Missing Attributes Values : None

    **Downloaded from :** https://www.kaggle.com/mathchi/diabetes-data-set  

    """)

    st.markdown("""
    ### 1.1. Data Dictionary

| Features       | description               |
| :------------- | :----------: -----------: |
| Pregnancies    | Number of times pregnant       |
| Glucose        | Plasma glucose concentration a 2 hours in an oral glucose tolerance test        |
| BloodPressure  | Diastolic blood pressure (mm Hg)       |
| SkinThickness  | Triceps skin fold thickness (mm)        |
| Insulin        | 2-Hour serum insulin (mu U/ml)     |
| BMI            | Body mass index (weight in kg/(height in m)^2)       |
| DiabetesPedigreeFunction   | Diabetes pedigree function      |
|Age     | Age (years)         |
|Outcome     | Class variable (0 or 1)         |

## 2. Exploratory Data Analysis

### 2.1. Data table

    """)


    df = pd.read_csv("data/diabetes.csv")
    st.dataframe(df)

    st.markdown("### 2.2. Descriptive Statistics")

    st.dataframe(df.describe())
    st.markdown("### 2.3. Features Variables Distribution")
    vdist = "images/variables_distribution.png"
    image = Image.open(vdist)
    st.image(image)

    st.markdown("### 2.4. Class Distribution")
    vdist = "images/class_distribution.png"
    image = Image.open(vdist)
    st.image(image)

    st.markdown("### 2.5. Correlation Matrix")
    vdist = "images/correlation_matrix.png"
    image = Image.open(vdist)
    st.image(image)

    st.markdown("### 2.6. Box Plot")
    vdist = "images/boxplot2.png"
    image = Image.open(vdist)
    st.image(image)

    st.markdown("""## 3. Data Cleaning

    Since nachine learning algorithm need clean data to produce a good result
    in this step of our project we took the following action to clean up data :
    * replace null values with the mean value of their columns
    * Set each value on their corresponding range accordingly to the domain of knowledge

    As result of data cleaning we two dataset was produce (clean_data.csv and scaled_data.csv)
    
    """)

    st.markdown("""## 4. Model
    """)

    st.markdown(""" ### 4.1. Model Selection""")

    vdist = "images/model_select.png"
    image = Image.open(vdist)
    st.image(image)

    st.markdown(""" 

    >  According to Area Under the Curve (AUC), Precision-Recall Curve and Accuracy 
    **Linear Discriminant Analysis** is the best model
    + **Hyperparameter :** 'solver': 'svd', 'store_covariance': True

    """)
    
    st.markdown(""" ### 4.1. Learning Curve""")

    vdist = "images/learning_curve.png"
    image = Image.open(vdist)
    st.image(image)

    st.markdown(""" ### 4.2. Confusion Matrix""")
   
    vdist = "images/confusion_matrix.png"
    image = Image.open(vdist)
    st.image(image)

    st.markdown(""" ### 4.3. Metric""")

    vdist = "images/metric.png"
    image = Image.open(vdist)
    st.image(image)

    st.markdown(""" ## 5. Room for Improvement
    
    
    To see the complete process with code of this project [clic Herte](https://github.com/RekidiangData-S/p01ml_diabetes_prediction)
    Feel free to improve or modify this project. I will be happy to have feedback from you.
    
    """)

  

# == All Result ===============================================================================================

def all_data():
    # Create your connection.
    cnx = sqlite3.connect('diabetes_result.db')

    df_all = pd.read_sql_query("SELECT * FROM patients", cnx)
    df_neg = pd.read_sql_query("SELECT * FROM patients WHERE Result = 'Negative'", cnx)
    df_pos = pd.read_sql_query("SELECT * FROM patients WHERE Result = 'Positive'", cnx)
    menu = ["All Result", "Positive Result", "Negative Result"]
    choice = st.radio("Menu", menu)

    if choice == "All Result":
        st.header("All Result")
        st.dataframe(df_all)
    elif choice == "Positive Result":
        st.header("Positive Result")
        st.dataframe(df_pos)
    elif choice == "Negative Result":
        st.header("Negative Result")
        st.dataframe(df_neg)


# == About ======================================================================
def about():
    st.markdown("""
    ### Motivation

    **Diabetes** is one of the diseases that affects many people in the world, detecting it early will allow effective 
   care taking of patient. This application  allows automatic and rapid prediction of diabetes in **prediabetic stage** 
   using certain symptom measurements.
   """)

    st.markdown("""
    ### Reading

    * [ papers that cite this data set](https://archive.ics.uci.edu/ml/support/Diabetes)
    * [PIMA Dataset and for Diabetes Analysis Review](https://ijsret.com/wp-content/uploads/2021/05/IJSRET_V7_issue3_495.pdf)
   """)

    st.markdown("""
    ### author

    I'm a data and technology passionate person, Artificial Intelligence enthusiast. For more information about me go to my **Website** and **Social Network** platform (👇)
    """)

# == Footer ==========================================================================================
def footer():
    footerr = """
            <div style="background-color:blue;padding:1px">
            <h4 style="color:white;text-align:center;">Kiese Diangebeni Reagan </h4>
            <p style="color:red;text-align:center;"> = Datalogue Date Passionate =</p>
            <p style="color:white;text-align:center;">www.kiesediangebeni.com, 
            E-mail : contact@kiesediangebeni.com</p>
            
            <p style="color:black;text-align:center;">
            <a href="https://twitter.com/ReaganKiese">Twitter</a> - 
            <a href="">Linkedin</a> - 
            <a href="https://github.com/RekidiangData-S">Github</a> - 
            <a href="https://medium.com/@rkddatas">Medium</a> - 
            <a href="https://www.kaggle.com/rekidiang">Kaggle</a></p>
            </div><br>"""
    st.markdown(footerr, unsafe_allow_html=True)
    st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)
