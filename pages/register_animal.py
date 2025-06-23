import streamlit as st
from services.database import connection

st.set_page_config(page_title="Register Mice", page_icon="🐭", layout="wide")

st.title("Register Mice")
st.write("Register a new mice in the database")

study_id = st.text_input("Identification")
sex = st.selectbox("sex", ['M', 'F'])
age = st.number_input("Age (days)", min_value=0, max_value=100, value=0)
weight = st.number_input("Weight", min_value=0.0, max_value=100.0, value=0.0, format="%.2f")
specie = st.selectbox("Species", ['mice', 'rat'])

if st.button("Register"):
    if study_id.strip() == "":
        st.error("Please enter a valid ID")
    else:
        connection.register_animal(study_id, specie, sex, age, weight)
        st.success("Mice registered successfully")

st.subheader("Registered Mice")
animals = connection.list_animals()

if animals is None or len(animals) == 0:
    st.info("No animals registered yet.")
else:
    st.dataframe(animals, use_container_width=True)
    # st.download_button(
    #     label="Download CSV",
    #     data=animals,
    #     file_name='animals.csv',
    #     mime='text/csv',
    #     key='download-csv'
    # )