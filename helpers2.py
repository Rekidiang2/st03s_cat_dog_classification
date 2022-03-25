import streamlit as st
import numpy as np
from PIL import Image
from PIL import Image, ImageOps
import pandas as pd
import keras
#import cv2 as cv
import pickle
import sqlite3
from utilities import teachable_machine_classification
#import matplotlib as plt

# == Logo
def logo():
    # source: jason-leung from unsplash
    logo = "images/ktlogo3.png"
    logo = Image.open(logo)
    size=(100,100)
    #resize image
    logo = logo.resize(size)
    st.sidebar.image(logo)

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)
    
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return prediction, np.argmax(prediction) # return position of the highest probability




# == Home =======================================================================================
def home():
    #st.text("Please Upload Image")
    uploaded_file = st.file_uploader("Dog or Cat Image ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    #using st.beta_columns.
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image(image, caption='Image Uploaded .', use_column_width=True)
        except UnboundLocalError: 
            st.text ("Waiting for an image ... ")
        else:
            st.text("Loaded")
    with col2:
        if st.button("Classify"):
            pred, label = teachable_machine_classification(image, 'models/keras_model.h5')
            if label == 0:
                st.success("It's a Dog !")
                score = f" I'm {round((100*(pred[0][0]),100*pred[0][0])[0],2)}% Sure"
                st.success(score)
                
            else:
                st.success("It's a Cat !")
                score = f" I'm {round((100*(1-pred[0][0]),100*pred[0][0])[0],2)}% Sure"
                st.success(score)


def galerie():
    st.title("Beautiful Cat and Dog")
    cat1 = Image.open("./images/cat1.jpg")
    cat2 = Image.open("images/cat2.jpg")
    cat3 = Image.open("images/cat3.jpg")
    cat4 = Image.open("images/cat4.jpg")

    dog1 = Image.open("images/dog1.jpg")
    dog2 = Image.open("images/dog2.jpg")
    dog3 = Image.open("images/dog3.jpg")
    dog4 = Image.open("images/dog4.jpg")


    col3, col4, col5, col6 = st.columns(4)
    dol3, dol4, dol5, dol6 = st.columns(4)
    fol3, fol4, fol5, fol6 = st.columns(4)

    col3.image(cat1, caption='Miao Miao', use_column_width=True)
    col4.image(dog1, caption='Chaka', use_column_width=True)
    col5.image(cat3, caption='Lupemba', use_column_width=True)
    col6.image(dog2, caption='Mashakado', use_column_width=True)
    dol3.image(cat2, caption='Lukas', use_column_width=True)
    dol4.image(dog3, caption='Lucio', use_column_width=True)
    dol5.image(cat4, caption='Lumbe Lumbe', use_column_width=True)
    dol6.image(dog4, caption='Ntela Ntela', use_column_width=True)


def training_info():
    
    #cover = Image.open('D:/rekidiang_DS/DS_projects/P03cv_cat_dog_classif_rkd/accessoirs/cover_img.PNG')
    #st.image(cover, use_column_width=True)
    #st.subheader("1) Data")
    st.markdown(""" 
    ## Data
    ---
    
    """)
    p1, p2,  = st.beta_columns(2)
    p1.success("class balance bar")
    p2.success("class balance pie")

    st.markdown(""" 
    ## Model
    ---
    
    
    """)
    st.markdown(""" ### Structure """)
    #str = Image.open('D:/rekidiang_DS/DS_projects/P03cv_cat_dog_classif_rkd/analysis-and_training/output/model_plot.png')
    #st.image(str, use_column_width=True)

    st.markdown(""" ### Evaluation """)

    v1, v2,  = st.beta_columns(2)
    v1.success("Accuracy plot")
    v2.success("Loss Plot")
    v11, v21,  = st.beta_columns(2)
    v11.success("Confusion Matrix")
    v21.success("Report")



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

    I’m  Data and technology passionate person, Artificial Intelligence enthusiast, lifelong learner. Since my childhood I was interested to technology and science, but I didn’t get access to it, by the lack of resource and opportunities hopefully grace to massive learning resource available on the Internet I’m getting close to my dream. My pleasure is to motivate, guide and teach people with less or without resource accomplish their dream in the world of technology specially kids and young. For more information about me go to my **Website** and **Social Network** platform (👇)
    """)

# == Footer ==========================================================================================
def footer():
    footerr = """
            ---
            ---
            <div style="background-color:white;padding:1px">
            <p style="color:blue;text-align:center;">Kiese Diangebeni Reagan </p>
            <p style="color:red;text-align:center;"> = Data Science Analyst =</p>
            <p style="color:blue;text-align:center;"><a href:"https://kiese.tech>www.kiese.tech</a></p>

            
            <p style="color:red;text-align:center;">
            <a href="https://twitter.com/ReaganKiese">Twitter</a> - 
            <a href="">Linkedin</a> - 
            <a href="https://github.com/RekidiangData-S">Github</a> - 
            <a href="https://medium.com/@rkddatas">Medium</a> - 
            <a href="https://www.kaggle.com/rekidiang">Kaggle</a></p>
            </div>"""
    st.markdown(footerr, unsafe_allow_html=True)
    st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)





