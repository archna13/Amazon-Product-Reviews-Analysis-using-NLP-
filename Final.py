import streamlit as st
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import easyocr
from PIL import Image
import pandas as pd
import numpy as np
import re
import pymysql
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image,ImageEnhance,ImageFilter,ImageOps,ImageDraw
import easyocr
from joblib import load
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")
import eda,nlp,product

#python -m streamlit run Final_proj_streamlit.py

# SETTING PAGE CONFIGURATIONS
icon = Image.open("istockphoto-1206796363-612x612.jpg")
st.set_page_config(page_title="Final Project:| By ARCHANA ",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """# This web application is created to the model prediction, price prediction, Image processing and NLP *!"""})
st.markdown("<h1 style='text-align: center; color: Green;",
            unsafe_allow_html=True)

#st.snow
#python -m streamlit run Final_proj_streamlit.py


# CREATING OPTION MENU
selected = option_menu(None, ["Home", "Customer_conversion","EDA" ,"Product_recommendation","NLP","Image"],
                       icons=["house", "cloud-upload", "pencil-square","gear","list-task"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "35px", "text-align": "centre", "margin": "-3px",
                                            "--hover-color": "#545454"},
                               "icon": {"font-size": "35px"},
                               "container": {"max-width": "6000px"},
                               "nav-link-selected": {"background-color": "#ff5757"}})

# HOME MENU
if selected == "Home":
    col1, col2 = st.columns(2)
    with col1:
        st.video("machine.mp4")
        st.markdown("## :green[**Technologies Used :**] Machine Learning,Python,easy OCR, Streamlit,Pandas,nltk,NER,plotly")
    with col2:
        st.write(
            '## This project is the comination of Machine Learning models, NLP, Complete EDA process and Image processing.:green[**Customer conversion models**] are commonly used in e-commerce, digital marketing, and online platforms to optimize conversion rates, personalize user experiences, and improve overall customer engagement.:green[**EDA involves**] using various statistical and graphical methods such as histograms, scatter plots, box plots, and correlation matrices to understand the underlying patterns and relationships within the data.Common techniques for product recommendation include collaborative filtering, content-based filtering, and hybrid methods that combine both approaches.:green[**Recommendation systems**] are widely used in e-commerce, streaming services, and online platforms.:green[**NLP**] is used in various applications, including sentiment analysis, chatbots, machine translation, text summarization, and information extraction. It enables machines to process and comprehend human language, making it valuable in many fields')


# Customer conversion
data = pd.read_csv("classification_data.csv")

# Load the models inside the Streamlit app
if selected == "Customer_conversion":
    col1, col2 = st.columns(2)
    with col1:
    # Load the Decision Tree model
        with open('class model.pkl', 'rb') as model_file:
             logistic_model = pickle.load(model_file)

        channelgrouplist = list(data['channelGrouping'].unique())
        channelgrouplist.sort()
        devices = list(data['device_deviceCategory'].unique())
        devices.sort()
        regions = list(data['geoNetwork_region'].unique())
        regions.sort()
        sources = list(data['latest_source'].unique())
        sources.sort()
        keyword = list(data['latest_keyword'].unique())
        keyword.sort()
        product_arr = list(data['products_array'].unique())
        product_arr.sort()

        device_deviceCategory = st.selectbox("Select Device Category", devices)
        geoNetwork_region = st.selectbox("Select GeoNetwork Region", regions)
        historic_session = st.number_input("Enter historic_session", min_value=0, value=0)
        historic_session_page = st.number_input("Enter historic_session_page", min_value=0, value=0)
        avg_session_time = st.number_input("Enter avg_session_time", min_value=0, value=0)
        avg_session_time_page = st.number_input("Enter avg_session_time_page", min_value=0, value=0)
        single_page_rate = st.number_input("Enter single_page_rate", min_value=0, value=0)
        sessionQualityDim = st.number_input("Enter sessionQualityDim", min_value=0, value=0)
        latest_visit_id = st.number_input("Enter latest_visit_id", min_value=0, value=0)
        latest_visit_number = st.number_input("Enter latest_visit_number", min_value=0, value=0)
        time_latest_visit = st.number_input("Enter time_latest_visit", min_value=0, value=0)
        avg_visit_time = st.number_input("Enter avg_visit_time", min_value=0, value=0)
        visits_per_day = st.number_input("Enter visits_per_day", min_value=0, value=0)
        latest_source = st.selectbox("Select Latest Source",sources)
        latest_medium = st.selectbox("Select Latest Medium", data['latest_medium'].unique())
        latest_keyword = st.selectbox("Enter Latest Keyword",keyword)
        latest_isTrueDirect = st.checkbox("Is True Direct", value=False)
        time_on_site = st.number_input("Enter time_on_site", min_value=0, value=0)
        products_array = st.selectbox("Enter product array",product_arr)
        transactionRevenue = st.number_input("Enter transactionRevenue", min_value=0, value=0)
        count_hit = st.number_input("Enter counthit")
        channelGrouping = st.selectbox("Enter channelGrouping",channelgrouplist)

        channels = int(channelgrouplist.index(channelGrouping))
        device = int(devices.index(device_deviceCategory))
        region = int(regions.index(geoNetwork_region))
        source = int(sources.index(latest_source))
        keywords = int(keyword.index(latest_keyword))
        product_arrr = int(product_arr.index(products_array))
    # Additional feature dictionary
    additional_feature = {
        "count_hit": count_hit,
        'channelGrouping':channels,
        'device_deviceCategory':device,
        'geoNetwork_region':region,
        'historic_session':historic_session,
        'historic_session_page':historic_session_page,
        'avg_session_time':avg_session_time,
        'avg_session_time_page':avg_session_time_page,
        'single_page_rate':single_page_rate,
        'sessionQualityDim':sessionQualityDim,
        'latest_visit_id':latest_visit_id,
        'latest_visit_number':latest_visit_number,
        'time_latest_visit':time_latest_visit,
        'avg_visit_time':avg_visit_time,
        'visits_per_day':visits_per_day,
        'latest_source':source,
        'latest_keyword':keywords,
        'latest_isTrueDirect':latest_isTrueDirect,
        'time_on_site':time_on_site,
        'transactionRevenue':transactionRevenue,
        'products_array':product_arrr
    }

    # Assuming all 21 features used during training are numerical
    all_features = [
            'count_hit', 'channelGrouping', 'device_deviceCategory',
        'geoNetwork_region', 'historic_session', 'historic_session_page',
        'avg_session_time', 'avg_session_time_page', 'single_page_rate',
        'sessionQualityDim', 'latest_visit_id', 'latest_visit_number',
        'time_latest_visit', 'avg_visit_time', 'visits_per_day',
        'latest_source', 'latest_keyword', 'latest_isTrueDirect',
        'time_on_site', 'transactionRevenue', 'products_array',  
        ]

    # Prediction button
    if st.button("Predict Conversion"):
        dff = pd.DataFrame([additional_feature])
        #dff = dff.apply(zscore)
        st.dataframe(dff)
        dt = logistic_model.predict(dff)

        st.write(dt)
        if  dt[0] == 0:
            st.write("Not converted")
        else:
            st.write("Converted")

#EDA
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# streamlit run streamlitEDAapp.py
if selected == "EDA":
    st.title('EDA Application')

    st.sidebar.title('Dataset Options')

    # uploader button
    st.sidebar.subheader('File Upload')
    uploaded_file = st.sidebar.file_uploader("classification_data.csv")


    # display rest of the application and options when a file is uploaded
    if uploaded_file is not None:

        # save the file to a pandas dataframe
        df = pd.read_csv(uploaded_file)
        

        # show dataframe info
        st.subheader('Basic Info:')
        st.write('There are ' + str(len(df.columns)) + ' columns in your dataset.')
        st.write('There are ' + str(len(df.index)) + ' rows in your dataset.')
        # Display summary statistics
        st.write(df.describe())
        # display the number of different types of variables
        datatypes = df.dtypes.value_counts() 
        for i in datatypes.index:
            st.write('The number of ', i, ' objects is ', datatypes[i])
        st.divider()


        



        # display dataframe for user to view on the main page, optional slider for displayed rows
        display_df = st.sidebar.checkbox('Preview Dataframe', key = 'disabled')
        if display_df:
            st.subheader(uploaded_file.name[0:-4] + ' Dataframe:')
            displayed_rows = st.sidebar.slider('How many rows do you want to see?', 1, len(df.index))
            st.dataframe(df.head(displayed_rows))
            st.divider()
        





        # set up column selection
        st.sidebar.divider()
        st.sidebar.subheader('Column Selection')
        col_type = st.sidebar.selectbox('Column Type', ('Numeric', 'Categorical'))



        # numeric column case + graph and five number summary
        if col_type == "Numeric":
            numeric_col = st.sidebar.selectbox('Column Name', df.select_dtypes(include = ['int64', 'float64']).columns)
            st.subheader(numeric_col + ' Info:')
            # five number summary
            df[numeric_col].describe().loc[['min', '25%', '50%', '75%', 'max']]

            st.divider()


            # histogram
            st.sidebar.divider()
            st.subheader('Histogram:')
            fig = plt.figure(figsize = (9, 7))

            # customize histogram options
            st.sidebar.subheader('Graph Options')
            hist_title = st.sidebar.text_input('Set Title', ('Histogram of ' + numeric_col))
            hist_xtitle = st.sidebar.text_input('Set X-Axis Title', numeric_col)
            hist_type = st.sidebar.selectbox('Type of histogram', ('count', 'frequency', 'density', 'probability', 'percent'))
            hist_kernel_density = st.sidebar.checkbox('Kernel Density Estimate')
            hist_bins = st.sidebar.slider('Number of Bins', min_value = 5, max_value = 100, value = 30)
            hist_color = st.sidebar.color_picker('Pick a color', '#4E99F1')
            hist_opacity = st.sidebar.slider('Bar Opacity', min_value = 0.0, max_value = 1.0, value = .5, step = .05)

            # create the histogram based on the specifications
            sns.histplot(data = df, x = numeric_col, stat = hist_type, color = hist_color, bins = hist_bins, alpha = hist_opacity, kde = hist_kernel_density)
            plt.title(hist_title)
            plt.xlabel(hist_xtitle)
            st.pyplot(fig)


            # saving the image
            filename = hist_title + '.png'
            fig.savefig(filename, dpi = 300)
            with open(filename, 'rb') as file:
                btn = st.download_button(
                label = 'Download Image',
                data = file,
                file_name = filename,
                mime = 'image/png'
                )


        # categorical column stuff and graph
        if col_type == "Categorical":
            cate_col = st.sidebar.selectbox('Select a column', df.select_dtypes(include = ['object']).columns)
            st.sidebar.divider()
            
            # make the proportion table
            st.subheader(cate_col + ' Info:')
            df[cate_col].value_counts() / len(df)
            st.divider()


            # barplot
            st.subheader('Graph:')
            fig = plt.figure(figsize = (9, 7))

            # customize barplot options
            st.sidebar.subheader('Graph Options')
            bar_y_axis = st.sidebar.selectbox('Y-Axis', df.select_dtypes(include = ['int64', 'float64']).columns)
            bar_title = st.sidebar.text_input('Set Title', ('Barplot of ' + cate_col))
            bar_xtitle = st.sidebar.text_input('Set X-Axis Title', cate_col)
            bar_ytitle = st.sidebar.text_input('Set Y-Axis Title', bar_y_axis)
            bar_opacity = st.sidebar.slider('Bar Opacity', min_value = 0.0, max_value = 1.0, value = 1.0, step = .05)
            countplot = st.sidebar.checkbox('Countplot')

            # create barplot using the specifications
            if not countplot:
                sns.barplot(data = df, x = cate_col, y = bar_y_axis, alpha = bar_opacity)
                plt.title(bar_title)
                plt.xlabel(bar_xtitle)
                plt.ylabel(bar_ytitle)
                st.pyplot(fig)
            else:
                sns.countplot(data=df, x = cate_col, alpha = bar_opacity)
                plt.title(bar_title)
                plt.xlabel(bar_xtitle)
                st.pyplot(fig)

            # saving the image
            filename = bar_title + '.png'
            fig.savefig(filename, dpi = 300)
            with open(filename, 'rb') as file:
                btn = st.download_button(
                label = 'Download Image',
                data = file,
                file_name = filename,
                mime = 'image/png'
                )
        
       

        



    
    # this is all commented out because for some reason streamlit's .map function was not displaying correctly
    # i checked the forums, but could not find a solution
    # this is possibly because mapbox's third party tos may have changed
    # if i find out how to fix it, make sure to reenable 'coordinates' as an option for column type


    # if col_type == "Coordinate":
    #     coor_col = st.sidebar.multiselect('Latitude and Logitude Columns', df.select_dtypes(include = ['int64', 'float64']).columns)
    #     st.sidebar.markdown('**You must put *latitude* first, and *longitude* second!**')
    #     st.sidebar.markdown('**Any non-coordinate columns will result in an error or faulty map!**')

    #     if coor_col:
    #         st.subheader('Point Map')
    #         map_df = df[[coor_col[0], coor_col[1]]].copy()
    #         map_df = map_df.dropna()
    #         st.map(map_df)

#Product_Recommendation

import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

if selected == "Product_recommendation":

    # Load the dataset
    data = pd.read_csv('amazon_product.csv')

    # Remove unnecessary columns
    data = data.drop('id', axis=1)

    # Define tokenizer and stemmer
    stemmer = SnowballStemmer('english')
    def tokenize_and_stem(text):
        tokens = nltk.word_tokenize(text.lower())
        stems = [stemmer.stem(t) for t in tokens]
        return stems

    # Create stemmed tokens column
    data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

    # Define TF-IDF vectorizer and cosine similarity function
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
    def cosine_sim(text1, text2):
        # tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
        text1_concatenated = ' '.join(text1)
        text2_concatenated = ' '.join(text2)
        tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
        return cosine_similarity(tfidf_matrix)[0][1]

    # Define search function
    def search_products(query):
        query_stemmed = tokenize_and_stem(query)
        data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
        results = data.sort_values(by=['similarity'], ascending=False).head(10)[['Title', 'Description', 'Category']]
        return results

    # web app
    img = Image.open('ima.jpg')
    st.image(img,width=600)
    st.title("Search Engine and Product Recommendation System ON Amazon Data")
    query = st.text_input("Enter Product Name")
    sumbit = st.button('Search')
    if sumbit:
        res = search_products(query)
        st.write(res)


import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import easyocr
import numpy as np
# Image processing

if selected == "Image":
    
    def image():
        st.write("<h4 style='text-align:center; font-weight:bolder;'>Image Processing</h4>", unsafe_allow_html=True)
        upload_file = st.file_uploader('Choose a Image File', type=['png', 'jpg', 'webp'])

        if upload_file is not None:
            upload_image = np.asarray(Image.open(upload_file))
            u1 = Image.open(upload_file)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Read Original Image")
                st.image(upload_image,)
                width = st.number_input("**Enter Width**", value=(u1.size)[0])

            with col2:
                graysclae = u1.convert("L")
                st.subheader("Gray Scale Image")
                st.image(graysclae)
                height = st.number_input("**Enter Height**", value=(u1.size)[1])

        # Continue with the rest of your image processing logic...
            with col1:
                resize_image = u1.resize((int(width), int(height)))
                st.subheader("Resize Image")
                st.image(resize_image)
                radius = st.number_input("**Enter radius**", value=1)
                blur_org = u1.filter(ImageFilter.GaussianBlur(radius=int(radius)))
                st.subheader("Blurring with Original Image")
                st.image(blur_org)
                blur_gray = graysclae.filter(ImageFilter.GaussianBlur(radius=int(radius)))
                st.subheader("Blurring with Gray Scale Image")
                st.image(blur_gray)
                threshold = st.number_input("**Enter Threshold**", value=100)
                threshold_image = u1.point(lambda x: 0 if x < threshold else 255)
                st.subheader("Threshold Image")
                st.image(threshold_image)
                flip = st.selectbox("**Select Flip**", ["left-right", 'top-bottom'])
                st.subheader("Flipped Image")
                if flip == "left-right":
                    st.image(u1.transpose(Image.FLIP_LEFT_RIGHT))
                if flip == 'top-bottom':
                    st.image(u1.transpose(Image.FLIP_TOP_BOTTOM))
                brightness = st.number_input("**Enter Brightness**", value=1)
                st.subheader("Brightness Image")
                st.image((ImageEnhance.Brightness(u1)).enhance(int(brightness)))

            with col2:
                mirror_image = ImageOps.mirror(u1)
                st.subheader("Mirror Image")
                st.image(mirror_image)
                contrast = st.number_input("**Enter contrast**", value=1)
                contrast_org = ImageEnhance.Contrast(blur_org)
                st.subheader("Contrast with Original Image")
                st.image(contrast_org.enhance(int(contrast)))
                contrast_gray = ImageEnhance.Contrast(blur_gray)
                st.subheader("Contrast with Gray Scale Image")
                st.image(contrast_gray.enhance(int(contrast)))
                rotation = st.number_input("**Enter Rotation**", value=180)
                st.subheader("Rotation Image")
                st.image(u1.rotate(int(rotation)))
                sharpness = st.number_input("**Enter Sharness**", value=1)
                st.subheader("Sharpness Image")
                st.image((ImageEnhance.Sharpness(u1)).enhance(int(sharpness)))
                image_type = st.selectbox("**Select Image**", ["Original image", 'Gray Scale Image', "Blur Image",
                                                            "Threshold Image", "Sharpness Image", "Brightness Image"])

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
            if bounds:
                st.subheader("Extracted Text")
                file_name = upload_file.name
                if file_name == '1.png':

                    address, city = map(str, (bounds[6][1]).split(', '))
                    state, pincode = map(str, (bounds[8][1]).split())
                    image1_data = {
                        'Company': bounds[7][1] + ' ' + bounds[9][1],
                        'Card_holder_name': bounds[0][1],
                        'Desination': bounds[1][1],
                        'Mobile': bounds[2][1],
                        'Email': bounds[5][1],
                        'URL': bounds[4][1],
                        'Area': address[0:-1],
                        'City': city[0:-1],
                        'State': state,
                        'Pincode': pincode
                    }
                    st.json(image1_data)

            # Continue with other conditions...
    image()

import streamlit as st
from transformers import pipeline
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import os
import spacy
from spacy import displacy

if selected == "NLP":
    # Download NLTK resources
    nltk.download('stopwords')

    # Streamlit app
    def main():
      st.title("Sentiment Analysis App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader("Raw Data")
        st.write(data)
        # Tokenize and calculate word frequency
        st.subheader("Word Frequency Analysis")

        # Tokenization
        text = " ".join(data["verified_reviews"].dropna())
        words = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

        # Frequency distribution
        freq_dist = FreqDist(words)

        # Display word frequency table
        st.write("Word Frequency Table:")
        st.write(pd.DataFrame(list(freq_dist.items()), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False))

        # Display word cloud
        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq_dist)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Function for lemmatization
    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)]
        return " ".join(lemmatized_words)

    # Streamlit app
    def main():
        st.title("Lemmatization in Streamlit")

        # User input
        user_input = st.text_area("Enter your text here")

        # Lemmatization button
        if st.button("Lemmatize"):
            with st.spinner("Lemmatizing..."):
                # Perform lemmatization
                lemmatized_text = lemmatize_text(user_input)

            # Display results
            st.write("## Original Text:")
            st.write(user_input)
            st.write("## Lemmatized Text:")
            st.write(lemmatized_text)

    if __name__ == "__main__":
        main()
    # Download spaCy model if not already downloaded
    spacy.cli.download("en_core_web_sm")

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Function for keyword extraction
    def extract_keywords(text):
        doc = nlp(text)
        keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
        return keywords

    # Streamlit app
    def main():
        st.title("Keyword Extraction App")

        # User input
        user_input = st.text_area("Enter your text here",key="user_input")

        # Keyword extraction button
        if st.button("Extract Keywords"):
            with st.spinner("Extracting Keywords..."):
                # Perform keyword extraction
                keywords = extract_keywords(user_input)

            # Display results
            st.write("## Original Text:")
            st.write(user_input)
            st.write("## Extracted Keywords:")
            st.write(", ".join(keywords))

    if __name__ == "__main__":
        main()   

    # Load spaCy model for NER
    nlp = spacy.load("en_core_web_sm")

    # Load NLTK's SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    def analyze_entities(text):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def analyze_sentiment(text):
        scores = sia.polarity_scores(text)
        compound_score = scores['compound']

        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def visualize_entities(text):
        doc = nlp(text)
        displacy.render(doc, style="ent", jupyter=False)

    def main():
        st.title("NER and Sentiment Analysis ")

        # Get user input
        user_input = st.text_area("Enter your text here", height=200)

        if st.button("Analyze Entities and Sentiment"):
            with st.spinner('Processing your text ...'):
                # Perform NER
                entities = analyze_entities(user_input)

                # Perform Sentiment Analysis
                sentiment = analyze_sentiment(user_input)

                # Display the Named Entities
                st.write("Named Entities:")
                for entity, label in entities:
                    st.write(f"{entity} - {label}")

                # Display the Sentiment Analysis result
                st.write(f"Sentiment Analysis: {sentiment}")

                # Visualize entities using spaCy's displaCy
                visualize_entities(user_input)

    if __name__ == "__main__":
        main()
