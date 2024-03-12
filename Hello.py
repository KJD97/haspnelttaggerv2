# Import necessary libraries
import streamlit as st
import pandas as pd
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to set up the Streamlit page
def do_stuff_on_page_load():
    st.set_page_config(layout="wide")

# Function to define NLTK syntactic tags
def define_syntactic_tags():
    st.markdown('# For each of the NLTK syntactic tags, define your own syntactic tag:')
    nltkTags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.', ',', '$']
    corpusTags = ['Con', 'Q', 'D', 'Ex', 'Fw', 'P', 'A', 'A', 'A', 'Ls', 'Mod', 'N', 'N', 'N', 'N', 'PDT', 'Pos', 'Pron', 'PosPro', 'Adv', 'Adv', 'Adv', 'RP', 'To', 'Uh', 'V', 'V', 'V', 'V', 'V', 'V', 'WhD', 'WhPron', 'WhPos', 'WhAdv', '.', ',', '$']
    tag_mapping = {}
    for i in range(len(nltkTags)):
        tag_mapping[nltkTags[i]] = st.text_input(nltkTags[i], corpusTags[i])
    return tag_mapping

# Function to define utterances or upload a .csv file
def define_utterances():
    st.markdown('### Step 2. Define your utterances or upload a .csv file with the following format ->')
    with open('data/utterances.csv') as f:
        st.download_button('Download Format CSV', f, 'utterances.csv', key='download_format_csv')

# Function to process data
def process_data(data, tag_mapping):
    st.markdown('### Step 3. Push the button to see and download results.')
    if st.button('Process'):
        utt = data.iloc[:, 0].values
        taggedUtt = []
        for u in utt:
            ut = ''
            text = word_tokenize(u)
            tags = nltk.pos_tag(text)
            for p in tags:
                if p[0] == ',':
                    ut += p[0] + ' '
                else:
                    ut += p[0] + '|' + tag_mapping[p[1]] + ' '
            taggedUtt.append(ut.strip())
        dft = pd.DataFrame({'utterance': utt, 'tagged': taggedUtt})
        st.dataframe(dft)

        # Save the processed data to a CSV file
        dft.to_csv('utterancesTagged.csv', index=False)
        st.download_button('Download CSV', 'utterancesTagged.csv')

# Main function
def main():
    do_stuff_on_page_load()

    st.markdown(f'''
    <style>
    .appview-container .main .block-container{{
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;    
    }}
    </style>
    ''', unsafe_allow_html=True)

    with st.sidebar:
        tag_mapping = define_syntactic_tags()

    st.title('HASPNeL Syntactic Tagger')
    st.markdown('***')

    st.markdown("The objective of this web app is to transform utterances in English as a list of strings into strings with the structure: `<word>|<category>` for each word of each utterance. The categorization is based on the Python's library, NLTK.")
    st.markdown('### Step 1. Open the sidebar on the left to define your own syntactic categories.')
    col1, col2 = st.columns([6, 2])
    with col1:
        col1.markdown('### Step 2. Define your utterances or upload a .csv file with the following format ->')
    with col2:
        define_utterances()

    option = st.selectbox('', ('Define', 'Upload'))

    if option == 'Define':
        # Initialize data
        data = pd.DataFrame({'utterance': []})

        def add_dfForm():
            row = pd.DataFrame({'utterance': [st.session_state.input_colA]})
            st.session_state.data = pd.concat([data, row], ignore_index=True)

        dfForm = st.form(key='dfForm')
        with dfForm:
            dfColumns = st.columns(1)
            with dfColumns[0]:
                st.text_area('Enter utterances to add them in the dataframe. Reload page to reset.', key='input_colA')
            st.form_submit_button(on_click=add_dfForm)

        st.dataframe(data)
    else:
        uploaded_file = st.file_uploader("Choose a file")
        try:
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.dataframe(data)
        except Exception as e:
            st.warning(f"Error: {e}")

    process_data(data, tag_mapping)

if __name__ == "__main__":
    main()

