import langchain_helper as lch
import streamlit as st

st.title("animal name generator")

user_animal_type = st.sidebar.selectbox("what is your animal?", ("","lion", "tiger"))
animal_color = ""
if user_animal_type=="tiger":
    animal_color = st.sidebar.text_area(label="what color is your tiger", max_chars=15)

if user_animal_type=="lion":
    animal_color = st.sidebar.text_area(label="what color is your tiger", max_chars=15)

if animal_color:
    response = lch.genereate_lion_name(user_animal_type, animal_color)
    st.text(response)