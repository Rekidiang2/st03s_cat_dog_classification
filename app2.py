import streamlit as st
from helpers2 import teachable_machine_classification
from PIL import Image, ImageOps
import keras

from helpers2 import logo, home, galerie, training_info, about, footer


# Sidebar Configuration
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#99ffcc,#99ffcc);
    color: purple;
}
</style>
""",
    unsafe_allow_html=True,
)

def main():
    
    logo()
    st.sidebar.title("Dag & Cat Classification")
    # basic layout
    menu = ["Home", "Dog & Cat Galerie", "Training Info", "About"]
    choice = st.sidebar.radio("Menu", menu)
    # siderbar method
    #st.write(dir(st.sidebar))

    html_temp = """
    <div style="background-color:blue;padding:0.5px">
    <h1 style="color:white;text-align:center;">Dog & Cat Classification </h1>
    </div><br>"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)

    if choice == "Home":

        #st.title("Home")
        home()
    elif choice == "Dog & Cat Galerie":
        st.subheader("Insert patient identity  and symptoms measurement")
        galerie()
    elif choice == "Training Info":
        st.header("Data Analysis and model trainig process")
        training_info()
    elif choice == "About":
        #st.title("About")
        about()


    footer()

if __name__ == '__main__':
    main()
