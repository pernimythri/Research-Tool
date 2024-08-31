from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import speech_recognition as sr
from datetime import datetime, timedelta, timezone
from transformers import pipeline

app = Flask(_name_)
app.secret_key = 'your_secret_key'  # Change this to a strong secret key

# Path to the CSV file
USER_FILE = 'users.csv'

# Load the QA pipeline with the specified model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Helper function to load users from CSV
def load_users():
    if os.path.exists(USER_FILE):
        return pd.read_csv(USER_FILE)
    else:
        return pd.DataFrame(columns=['Username', 'Password'])

def voice_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        print("Could not request results. Please check your internet connection.")
        return None

# Helper function to save users to CSV
def save_users(users):
    users.to_csv(USER_FILE, index=False)

# Function to perform web search
def search_web(query):
    try:
        query = quote_plus(query)  # URL encode the query
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []

        for g in soup.find_all('div', class_='tF2Cxc'):
            title_tag = g.find('h3')
            link_tag = g.find('a')
            description_tag = g.find('div', class_='VwiC3b')

            if title_tag and link_tag and description_tag:
                title = title_tag.text
                link = link_tag['href']
                description = description_tag.text
                results.append({"title": title, "link": link, "description": description})

        return results

    except requests.exceptions.RequestException as e:
        print(f"Error fetching search results: {e}")
        return []

# Function to process user input and get search results
def process_input(user_input, lines=5):
    search_results = search_web(user_input)
    if not search_results:
        return "Sorry, I couldn't find any relevant information."
    
    # Concatenate up to 'lines' lines of descriptions
    answer_lines = []
    for result in search_results[:lines]:  # Limit to 'lines' results
        answer_lines.append(result['description'])

    answer = "<br>".join(answer_lines)
    return f"<br>{answer}<br>Source: <a href='{search_results[0]['link']}' target='_blank'>{search_results[0]['link']}</a>"

def extract_text_from_url(url):
    """
    Extracts text content from a given URL using web scraping.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text content from the webpage
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    except Exception as e:
        print(f"Error extracting text from {url}: {str(e)}")
        return None

def answer_questions_from_urls(urls, questions):
    results = []

    for url in urls:
        text = extract_text_from_url(url)
        if text:
            for question in questions:
                try:
                    result = qa_pipeline(question=question, context=text)
                    results.append({
                        'url': url,
                        'question': question,
                        'answer': result['answer']
                    })
                except Exception as e:
                    print(f"Error answering question '{question}' for URL {url}: {str(e)}")

    return results

# Function to clear session history if older than 1 hour
def clear_old_history():
    session_timestamp = session.get('timestamp')
    if session_timestamp:
        session_age = datetime.now(timezone.utc) - datetime.fromisoformat(session_timestamp)
        if session_age > timedelta(hours=1):
            session.pop('history', None)
            session.pop('timestamp', None)

# Function to limit history to the most recent 3 entries
def limit_history(history):
    if len(history) > 3:
        return history[-3:]
    else:
        return history

# Routes

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['Username']
        password = request.form['Password']
        users = load_users()

        if username not in users['Username'].values:
            return render_template('login.html', message='Username does not exist')
        
        user_data = users[users['Username'] == username].iloc[0]
        if str(user_data['Password']) != password:
            return render_template('login.html', message='Incorrect password')
        
        session['username'] = username
        session['timestamp'] = datetime.now(timezone.utc).isoformat()

        if 'history' not in session or not isinstance(session['history'], dict):
            session['history'] = {}

        if username not in session['history']:
            session['history'][username] = []

        return redirect(url_for('home'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['Username']
        password = request.form['Password']
        users = load_users()

        if username in users['Username'].values:
            return render_template('register.html', message='Username already exists')
        
        new_user = pd.DataFrame([[username, password]], columns=['Username', 'Password'])
        users = pd.concat([users, new_user], ignore_index=True)
        save_users(users)
        
        return render_template('register.html', message=f'Successfully registered as {username}')
    
    return render_template('register.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    clear_old_history()  # Clear old history if older than 1 hour
    if request.method == 'POST':
        question = request.form['Question'].strip()
        urls_input = request.form['Urls'].strip()
        urls = [url.strip() for url in urls_input.split(',') if url.strip().startswith('http')]
        
        if urls:
            questions = [question]
            answers = answer_questions_from_urls(urls, questions)
            username = session['username']
            
            if 'history' not in session or not isinstance(session['history'], dict):
                session['history'] = {}

            if username not in session['history'] or not isinstance(session['history'][username], list):
                session['history'][username] = []

            for answer in answers:
                session['history'][username].append({'question': answer['question'], 'answer': answer['answer'], 'source': answer['url']})
            
            session['history'][username] = limit_history(session['history'][username])
            session['timestamp'] = datetime.now(timezone.utc).isoformat()
            session.modified = True
            return redirect(url_for('home'))
        
        else:
            response = process_input(question, lines=5)  # Return 5 lines of content
            username = session['username']
            
            if 'history' not in session or not isinstance(session['history'], dict):
                session['history'] = {}

            if username not in session['history'] or not isinstance(session['history'][username], list):
                session['history'][username] = []

            session['history'][username].append({'question': question, 'answer': response})
            session['history'][username] = limit_history(session['history'][username])
            session['timestamp'] = datetime.now(timezone.utc).isoformat()
            session.modified = True
            return redirect(url_for('home'))
    
    return render_template('home.html', history=session['history'].get(session['username'], []))

if _name_ == '_main_':
    app.run(host='0.0.0.0', debug=True)
