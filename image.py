import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image,ImageEnhance,ImageFilter,ImageOps,ImageDraw
import easyocr
from joblib import load
import plotly.express as px

def classification():
    pass

def eda():
    pass

def image():
    st.write("<h4 style='text-align:center;  font-weight:bolder;'>Image Processing</h4>",unsafe_allow_html=True)
    upload_file = st.file_uploader('Choose a Image File', type=['png','jpg','webp'])

    if upload_file is not None:
        upload_image = np.asarray(Image.open(upload_file))
        u1 = Image.open(upload_file)
        
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Read Original Image")
            st.image(upload_image,)
            width =st.number_input("**Enter Width**",value=(u1.size)[0])
        
        with col2:
            graysclae = u1.convert("L")                   
            st.subheader("Gray Scale Image")
            st.image(graysclae)
            height =st.number_input("**Enter Height**",value=(u1.size)[1])

        with col1:
            resize_image = u1.resize((int(width),int(height)))
            st.subheader("Resize Image")
            st.image(resize_image)
            radius =st.number_input("**Enter radius**",value=1)
            blur_org = u1.filter(ImageFilter.GaussianBlur(radius = int(radius) ))
            st.subheader("Blurring with Original Image")
            st.image(blur_org)
            blur_gray = graysclae.filter(ImageFilter.GaussianBlur(radius = int(radius) ))
            st.subheader("Blurring with Gray Scale Image")
            st.image(blur_gray)
            threshold = st.number_input("**Enter Threshold**",value=100)
            threshold_image = u1.point(lambda x: 0 if x< threshold else 255)
            st.subheader("Threshold Image")
            st.image(threshold_image)
            flip = st.selectbox("**Select Flip**",["left-right",'top-bottom'])
            st.subheader("Flipped Image")
            if flip == "left-right":
                st.image(u1.transpose(Image.FLIP_LEFT_RIGHT))
            if flip == 'top-bottom':
                st.image(u1.transpose(Image.FLIP_TOP_BOTTOM))
            brightness =st.number_input("**Enter Brightness**",value=1)
            st.subheader("Brightness Image")
            st.image((ImageEnhance.Brightness(u1)).enhance(int(brightness)))


        with col2:
            mirror_image = ImageOps.mirror(u1)
            st.subheader("Mirror Image")
            st.image(mirror_image)
            contrast =st.number_input("**Enter contrast**",value=1)
            contrast_org = ImageEnhance.Contrast(blur_org)
            st.subheader("Contrast with Original Image")
            st.image(contrast_org.enhance(int(contrast)))
            contrast_gray = ImageEnhance.Contrast(blur_gray)
            st.subheader("Contrast with Gray Scale Image")
            st.image(contrast_gray.enhance(int(contrast)))
            rotation =st.number_input("**Enter Rotation**",value=180)
            st.subheader("Rotation Image")
            st.image(u1.rotate(int(rotation)))
            sharpness =st.number_input("**Enter Sharness**",value=1)
            st.subheader("Sharpness Image")
            st.image((ImageEnhance.Sharpness(u1)).enhance(int(sharpness)))
            image_type = st.selectbox("**Select Image**",["Original image",'Gray Scale Image',"Blur Image","Threshold Image","Sharpness Image","Brightness Image"])
            
            if image_type == "Original image":
                st.subheader("Edge Detection with Original Image")
                st.image(u1.filter(ImageFilter.FIND_EDGES))
            if image_type == 'Gray Scale Image':
                st.subheader("Edge Detection with Grayscale Image")
                st.image(graysclae.filter(ImageFilter.FIND_EDGES))
            if image_type == "Blur Image":
                st.subheader("Edge Detection with Blur Original Image")
                st.image(blur_org.filter(ImageFilter.FIND_EDGES))
            
            if image_type == "Threshold Image":
                st.subheader("Edge Detection with Threshold Image")
                st.image(threshold_image.filter(ImageFilter.FIND_EDGES))
            if image_type == "Sharpness Image":
                st.subheader("Edge Detection with Sharpness Image")
                st.image(((ImageEnhance.Sharpness(u1)).enhance(int(sharpness))).filter(ImageFilter.FIND_EDGES))
            if image_type == "Brightness Image":
                st.subheader("Edge Detection with Brightness Image")
                st.image(((ImageEnhance.Brightness(u1)).enhance(int(brightness))).filter(ImageFilter.FIND_EDGES))
        
        reader = easyocr.Reader(['en'])
        bounds = reader.readtext(upload_image)
        if bounds != '':
            st.subheader("Extracted Text")
            file_name = upload_file.name
            if file_name == '1.png':
                
                address,city = map(str,(bounds[6][1]).split(', '))
                state,pincode = map(str,(bounds[8][1]).split())
                image1_data = {
                    'Company': bounds[7][1]+' '+bounds[9][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[2][1],
                    'Email': bounds[5][1],
                    'URL': bounds[4][1],
                    'Area':address[0:-1],
                    'City': city[0:-1],
                    'State':state,
                    'Pincode': pincode
                }
                st.json(image1_data)
                    
            if file_name == '2.png':
                state,pincode = map(str,(bounds[9][1]).split())
                image2_data = {
                    'Company': bounds[8][1]+' '+bounds[10][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[2][1],
                    'Email': bounds[3][1],
                    'URL': bounds[4][1]+'.'+bounds[5][1],
                    'Area': (bounds[6][1]+' '+bounds[11][1])[0:-2],
                    'City': (bounds[7][1])[0:-1],
                    'State':state,
                    'Pincode': pincode
                }
                st.json(image2_data)
                

            if file_name == '3.png':
                address,city = map(str,(bounds[2][1]).split(', '))
                state,pincode = map(str,(bounds[3][1]).split())
                image3_data = {
                    'Company': bounds[7][1]+' '+bounds[8][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[4][1],
                    'Email': bounds[5][1],
                    'URL': bounds[6][1],
                    'Area': address[0:-1],
                    'City': city[0:-1],
                    'State':state,
                    'Pincode': pincode
                }
                st.json(image3_data)


            if file_name == '4.png':
                area,city,state = map(str,(bounds[2][1]).split(', '))
                image4_data = {
                    'Company': bounds[6][1]+' '+bounds[8][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[4][1],
                    'Email': bounds[5][1],
                    'URL': bounds[7][1],
                    'Area': area[0:-1],
                    'City': city,
                    'State':state,
                    'Pincode': bounds[3][1]
                }
                st.json(image4_data)
                

            if file_name == '5.png':
                area,city,state = map(str,(bounds[2][1]).split(', '))
                image5_data = {
                    'Company': bounds[7][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[4][1],
                    'Email': bounds[5][1],
                    'URL': bounds[6][1],
                    'Area': area[0:-1],
                    'City': city,
                    'State':state,
                    'Pincode': bounds[3][1]
                }
                st.json(image5_data)
        

        
            

        
            

def nlp():
    pass

def recommendation():
    pass

st.set_page_config(layout='wide')

st.write("<h2 style='text-align:center; margin-top:-60px; font-weight:bolder; '>Final Project</h2>",unsafe_allow_html=True)

option = option_menu(
    menu_title=None,
    options=['CLASSIFICATION','EDA','IMAGE','NLP','RECOMMENDATION'],
    orientation='horizontal',
    styles={
                "container": {"padding": "0!important", "background-color": "#fafafa", "width": '100%', },
                "icon": {"color": "orange", "font-size": "11px"},
                "nav-link": {
                    "font-size": "11px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#525ceb",
                },
                "nav-link-selected": {"background-color": "green"},
            },
    )

st.markdown(
        """
        <style>
            .st-ax{
                background-color: lightblue;
                
            }

            
            .stNumberInput input{
                background-color: lightblue;
            }

            

        </style>
        """
    ,unsafe_allow_html=True
    )

if option == "CLASSIFICATION":
    classification()

if option == "EDA":
    eda()

if option == "IMAGE":
    image()

if option == "NLP":
    nlp()

if option == "RECOMMENDATION":
    recommendation() 