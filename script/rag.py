from dotenv import load_dotenv

import PyPDF2
import sqlite3

import numpy as np
import pickle

from openai import OpenAI
import os

import streamlit as st


def extract_text_from_pdf(pdf_file):
    '''
    This function extracts text from a PDF file.
    '''
    pdf_file = open(pdf_file, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text


def chunking(text):
    '''
    This function chunks the text into smaller pieces to be used for creating embeddings.
    Chunk size is 1000 and the overlap is 200.
    '''
    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    return chunks


def make_embeddings(client, chunks):
    '''
    This function creates embeddings for the chunks of text using the OpenAI API.
    '''
    
    def _make_embedding(client, chunk, model="text-embedding-3-small"):
        chunk = chunk.replace("\n", " ")
        return client.embeddings.create(input = [chunk], model=model).data[0].embedding
    
    embeddings = []
    for chunk in chunks:
        embedding = _make_embedding(client, chunk)
        embeddings.append(embedding)
    return embeddings
    
    
def create_database(database_name):
    '''
    This funciton creates a database to store the embeddings.
    Columns: id, text, embedding
    '''
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='embeddings'")
    if c.fetchone()[0] == 0:
        # If the table doesn't exist, create it
        c.execute('''CREATE TABLE embeddings
                     (id INTEGER PRIMARY KEY,
                     text TEXT,
                     embedding BLOB)''')
    conn.commit()
    conn.close()
    
    
def insert_embedding(database_name, text, embedding):
    '''
    This function inserts the text and its embedding into the database.
    '''
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    
    embedding_serialized = pickle.dumps(embedding)
    c.execute("INSERT INTO embeddings (text, embedding) VALUES (?, ?)", (text, embedding_serialized))
    conn.commit()
    conn.close()

def search_similar_text(database_name, query_embedding, num_results=5):
    '''
    This function performs the 
    '''
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    c.execute("SELECT text, embedding FROM embeddings")
    results = c.fetchall()
    conn.close()
    results = [(text, pickle.loads(embedding)) for text, embedding in results]
    
    # calculate the cosine similarity
    similarities = []
    for text, embedding in results:
        similarity = np.dot(query_embedding, embedding)/(np.linalg.norm(query_embedding)*np.linalg.norm(embedding))
        similarities.append((text, embedding, similarity))
    similarities.sort(key=lambda x: x[2], reverse=True)
    # get the top 5 similar texts
    return similarities[:num_results]


def get_response(client, system_content="", assistant_content="", user_content="", model="gpt-3.5-turbo"):
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": user_content}
        ],
        stream=True,
    )
    return chat_completion


def check_db_exists(database_name):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='embeddings'")
    if c.fetchone()[0] == 0:
        return False
    else:
        return True


if __name__ == '__main__':
    
    
    pdf_file = '../data/Arsenal_FC.pdf'
    
    text = extract_text_from_pdf(pdf_file)
    
    chunks = chunking(text)
    
    # Load the environment variables (API keys)
    load_dotenv()
    
    client = OpenAI()
    
    database_name = '../data/arsenal_embedding_database.db'
    
    if not check_db_exists(database_name):
        print("Embedding database does not exist. Creating one...")
        
        embeddings = make_embeddings(client, chunks)
    
        create_database(database_name)
    
        for chunk, embedding in zip(chunks, embeddings):
            insert_embedding(database_name, chunk, embedding)
            
        print("Database created.")
    else:
        print("Database already exists.")
    
    
    # frontend building
    st.title("Arsenal Football Club Chatbot - RAG")
    
    if "message" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            pass
    
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
    with st.chat_message("assistant"):
        if prompt:
            prompt_embedding = make_embeddings(client, [prompt])[0]
            context = search_similar_text(database_name, prompt_embedding)
            context = [text for text, embedding, similarity in context]
            
            chat_completion = get_response(
                client,
                system_content="",
                assistant_content="".join(context),
                user_content=prompt
            )
            response = st.write_stream(chat_completion)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        else:
            pass
        


    
        
    