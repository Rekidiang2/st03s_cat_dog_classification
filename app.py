import streamlit as st
from utilities import teachable_machine_classification
from PIL import Image, ImageOps
import toml

def main():
    # basic layout
    menu = ["Home", "Cat & Dog Beauty", "About"]
    choice = st.sidebar.radio("Menu", menu)
    #siderbar method
    #st.write(dir(st.sidebar))
    
    if choice == "Home":
        st.title("Cat or Dog")

        
        primaryColor = toml.load("config.toml")['theme']['primaryColor']
        s = f"""
        <style>
        div.stButton > button:first-child {{ border: 10px solid {primaryColor}; border-radius:50px 50px 50px 50px; }}
        <style>
        """
        st.markdown(s, unsafe_allow_html=True)

        #st.text("Please Upload Image")
        uploaded_file = st.file_uploader("Dog or Cat Image ...", type="jpg")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
       
        #using st.beta_columns.
        col1, col2 = st.beta_columns(2)
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
        
    elif choice == "Cat & Dog Beauty":
        st.title("Beautiful Cat and Dog")
        cat1 = Image.open("cat_dog_img/cat1.jpg")
        cat2 = Image.open("cat_dog_img/cat2.jpg")
        cat3 = Image.open("cat_dog_img/cat3.jpg")
        cat4 = Image.open("cat_dog_img/cat4.jpg")

        dog1 = Image.open("cat_dog_img/dog1.jpg")
        dog2 = Image.open("cat_dog_img/dog2.jpg")
        dog3 = Image.open("cat_dog_img/dog3.jpg")
        dog4 = Image.open("cat_dog_img/dog4.jpg")


        col3, col4, col5, col6 = st.beta_columns(4)
        dol3, dol4, dol5, dol6 = st.beta_columns(4)
        fol3, fol4, fol5, fol6 = st.beta_columns(4)

        col3.image(cat1, caption='Miao Miao', use_column_width=True)
        col4.image(dog1, caption='Chaka', use_column_width=True)
        col5.image(cat3, caption='Lupemba', use_column_width=True)
        col6.image(dog2, caption='Mashakado', use_column_width=True)
        dol3.image(cat2, caption='Lukas', use_column_width=True)
        dol4.image(dog3, caption='Lucio', use_column_width=True)
        dol5.image(cat4, caption='Lumbe Lumbe', use_column_width=True)
        dol6.image(dog4, caption='Ntela Ntela', use_column_width=True)

    else:
        st.title("Cat and Dog Classification")
        cover = Image.open('D:/rekidiang_DS/DS_projects/P03cv_cat_dog_classif_rkd/accessoirs/cover_img.PNG')
        st.image(cover, use_column_width=True)
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
        str = Image.open('D:/rekidiang_DS/DS_projects/P03cv_cat_dog_classif_rkd/analysis-and_training/output/model_plot.png')
        st.image(str, use_column_width=True)

        st.markdown(""" ### Evaluation """)

        v1, v2,  = st.beta_columns(2)
        v1.success("Accuracy plot")
        v2.success("Loss Plot")
        v11, v21,  = st.beta_columns(2)
        v11.success("Confusion Matrix")
        v21.success("Report")

        st.markdown(""" 
        ## About Me
        ---
         """)

        k1, k2, k3  = st.beta_columns(3)
        me = Image.open("D:/rekidiang_DS/DS_projects/P03cv_cat_dog_classif_rkd/accessoirs/me.jfif")
        k1.image(me, caption='Kiese Diangebeni Reagan', use_column_width=False)
        k2.markdown(""" 
        #### Kiese Diangebeni Reagan
      
        **Data Scientist, R & Python Developer** 

        Artificial Intelligence and Bioinformatic Enthusiast

        
       
         """)
        k3.markdown(""" 
        ### Follow Me on :

        [Linkedin](), 
        [Tweeter](), 
        [GitHub]()
       
         """)


if __name__ == '__main__':
    main()
