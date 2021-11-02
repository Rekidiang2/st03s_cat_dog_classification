import streamlit as st
from annex import home, app, analysis, all_data, about, footer

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
    # basic layout
    menu = ["Home", "Prediction", "Prediction Result",  "Analysis", "About"]
    choice = st.sidebar.radio("Menu", menu)
    # siderbar method
    #st.write(dir(st.sidebar))

    html_temp = """
    <div style="background-color:blue;padding:0.5px">
    <h1 style="color:white;text-align:center;">Diabetes Prediction </h1>
    </div><br>"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)

    if choice == "Home":

        #st.title("Home")
        home()
    elif choice == "Prediction":
        st.subheader("Insert patient identity  and symptoms measurement")
        app()
    elif choice == "Analysis":
        st.header("Data Analysis and model trainig process")
        analysis()
    elif choice == "Prediction Result":
        st.header("Prediction Result")
        all_data()
    elif choice == "About":
        #st.title("About")
        about()


    footer()

if __name__ == '__main__':
    main()
