# Code Debugger Integration for PyQnAsta Chatbot

import streamlit as st
import json
import numpy as np
import pickle
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load datasets
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

qa_data = load_json_data("Q&A.json")
pyqna_data = load_json_data("PyQnAsta.json")
explanation_data = load_json_data("explaination.json")

qa_answers = {entry["question"]: entry["answer"] for entry in qa_data}
pyqna_descriptions = {entry["title"]: entry["description"] for entry in pyqna_data}
pyqna_code = {entry["title"]: entry["code"] for entry in pyqna_data}
explain_texts = {entry["title"]: entry["explanation"] for entry in explanation_data}

all_texts = list(qa_answers.keys()) + list(pyqna_descriptions.keys()) + list(explain_texts.keys())
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Speech recognition
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        return ""

# Matching and response
def find_best_match(user_input):
    input_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    best_match_index = np.argmax(similarity_scores)
    best_match_text = all_texts[best_match_index]
    confidence = similarity_scores[0][best_match_index]
    return best_match_text, confidence

def suggest_questions(user_input):
    input_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][-5:][::-1]
    suggested = []
    for i in top_indices:
        question = all_texts[i]
        if not any(cosine_similarity(vectorizer.transform([question]), vectorizer.transform([q]))[0][0] > 0.8 for q in suggested):
            suggested.append(question)
        if len(suggested) == 3:
            break
    return suggested

def generate_response(user_input):
    if user_input.lower() in ["hi", "hello", "hey"]:
        return "Hello! How can I assist you with Python today?", []
    if user_input.lower() in ["bye", "thank you", "see you later"]:
        return "Goodbye! Have a great day!", []

    best_match, confidence = find_best_match(user_input)
    if confidence > 0.5:
        if best_match in qa_answers:
            return qa_answers[best_match], suggest_questions(user_input)
        if best_match in pyqna_descriptions:
            response_text = pyqna_descriptions[best_match]
            code_text = pyqna_code[best_match]
            return (response_text, code_text), suggest_questions(user_input)
        if best_match in explain_texts:
            return f"Explanation: {explain_texts[best_match]}", suggest_questions(user_input)

    return "I'm sorry, I couldn't understand your query. Could you please rephrase?", []

# Page config
st.set_page_config(page_title="PyQnAsta Chatbot", layout="wide")

# Session state
if "menu_open" not in st.session_state:
    st.session_state.menu_open = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'saved_chats' not in st.session_state:
    st.session_state.saved_chats = {}
if 'chat_title' not in st.session_state:
    st.session_state.chat_title = f"Chat {len(st.session_state.saved_chats) + 1}"
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []

# Header
col1, col2, col3 = st.columns([1, 15, 1])
with col1:
    st.image("logo.png", width=50)
with col2:
    st.markdown("<h5 style='color:black;'>PyQnAsta - Python Question and Answer Assistant</h5>", unsafe_allow_html=True)
with col3:
    if st.button("☰"):
        st.session_state.menu_open = not st.session_state.menu_open

# Sidebar
if st.session_state.menu_open:
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.subheader("Chat History")
        for title in st.session_state.saved_chats.keys():
            if st.sidebar.button(title):
                st.session_state.chat_history = st.session_state.saved_chats[title]
                st.session_state.chat_title = title
                st.rerun()
    with col2:
        if st.button("＋"):
            if st.session_state.chat_history:
                st.session_state.saved_chats[st.session_state.chat_title] = st.session_state.chat_history.copy()
                st.session_state.chat_title = f"Chat {len(st.session_state.saved_chats) + 1}"
                st.session_state.chat_history = []
                st.rerun()

# Custom chat CSS
st.markdown(""" <style>
.chat-bubble {
    padding: 12px 16px;
    border-radius: 18px;
    margin: 6px;
    max-width: 80%;
    line-height: 1.5;
    font-size: 16px;
    display: inline-block;
}
.user-bubble {
    background-color: black;
    color: white;
    align-self: flex-end;
    margin-left: auto;
}
.bot-bubble {
    background-color: #E5E5EA;
    color: black;
    align-self: flex-start;
    margin-right: auto;
}
.chat-container {
    display: flex;
    flex-direction: column;
}
.scroll-box {
    max-height: 500px;
    overflow-y: auto;
    padding-right: 10px;
} </style>
""", unsafe_allow_html=True)

# Welcome text
st.markdown("<h1 style='text-align: center; color: lightgrey;'>Welcome To PyQnAsta</h1>", unsafe_allow_html=True)

# Chat Display with bubbles
st.markdown("<div class='scroll-box'>", unsafe_allow_html=True)
with st.container():
    for chat in st.session_state.chat_history:
        if isinstance(chat, dict) and "code" in chat:
            st.code(chat["code"], language="python")
        elif chat.startswith("You:"):
            st.markdown(f"<div class='chat-container'><div class='chat-bubble user-bubble'>{chat[4:]}</div></div>", unsafe_allow_html=True)
        elif chat.startswith("PyQnAsta:"):
            st.markdown(f"<div class='chat-container'><div class='chat-bubble bot-bubble'>{chat[9:]}</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.suggested_questions:
    st.markdown("**Suggested Questions:**")
    for i, suggestion in enumerate(st.session_state.suggested_questions):
        if st.button(suggestion, key=f"suggestion_{i}"):
            response, _ = generate_response(suggestion)
            st.session_state.chat_history.append(f"You: {suggestion}")
            if isinstance(response, tuple):
                desc, code = response
                st.session_state.chat_history.append(f"PyQnAsta: {desc}")
                st.session_state.chat_history.append({"code": code})
            else:
                st.session_state.chat_history.append(f"PyQnAsta: {response}")
            st.session_state.suggested_questions = []
            st.rerun()

# Input Area
with st.container():
    user_input_col, send_col, mic_col = st.columns([14, 1, 2])
    with user_input_col:
        user_input = st.text_input("", placeholder="Type your message...", key="input", label_visibility="collapsed")
    with mic_col:
        if st.button("🎤"):
            voice_input = recognize_speech()
            if voice_input:
                st.session_state.input = voice_input
                st.rerun()
    with send_col:
        send_button = st.button("➤")

# Handle send
if send_button and st.session_state.input:
    response, suggested_questions = generate_response(st.session_state.input)
    st.session_state.chat_history.append(f"You: {st.session_state.input}")

    if isinstance(response, tuple):
        desc, code = response
        st.session_state.chat_history.append(f"PyQnAsta: {desc}")
        st.session_state.chat_history.append({"code": code})
    else:
        st.session_state.chat_history.append(f"PyQnAsta: {response}")

    st.session_state.suggested_questions = suggested_questions
    st.rerun()

# Code Debugger Section
# Code Debugger Section
if "show_debugger" not in st.session_state:
    st.session_state.show_debugger = False
if st.button("Open Code Debugger"):
    st.session_state.show_debugger = not st.session_state.show_debugger
def show_code_debugger_sidebar():
    import sys, io, traceback, datetime, time, re, ast
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
    from streamlit_ace import st_ace

    # --- Handle Reset BEFORE any widget is rendered ---
    if st.session_state.get("reset_debug_triggered", False):
        st.session_state["code_input"] = ""
        st.session_state["mock_input_sidebar"] = ""
        st.session_state["reset_debug_triggered"] = False
        st.rerun()

    with st.sidebar:
        st.subheader("Enhanced Code Debugger")

        # Theme selection
        theme = st.radio(" Theme", ["Light", "Dark"], key="debug_theme_sidebar")
        ace_theme = 'monokai' if theme == "Dark" else 'github'

        # Code editor
        code_input = st_ace(
            value=st.session_state.get("code_input", ""),
            language='python',
            theme=ace_theme,
            height=300,
            key="ace_editor_sidebar"
        )
        st.session_state["code_input"] = code_input

        # Input simulation if needed
        requires_input = "input(" in code_input
        mock_inputs = []
        if requires_input:
            st.info("Detected `input()` in code. Provide input values.")
            mock_input_text = st.text_area(" Input (one per line):", height=100, key="mock_input_sidebar")
            mock_inputs = [line.strip() for line in mock_input_text.strip().split('\n') if line.strip()]

        # --- RUN CODE button ---
        run_clicked = st.button("Run Code", key="run_debug_sidebar")

        # --- RESET CODE button ---
        reset_clicked = st.button("Reset Code", key="reset_debug_sidebar")

        # Handle Reset
        if reset_clicked:
            st.session_state["reset_debug_triggered"] = True
            st.rerun()

        # Run code logic
        if run_clicked:
            if not code_input.strip():
                st.warning("⚠ Please enter code.")
            else:
                try:
                    ast.parse(code_input)
                except SyntaxError as e:
                    st.error("❌ Syntax Error:")
                    st.code(str(e), language="python")
                else:
                    def run_code(code_input, mock_inputs):
                        output = io.StringIO()
                        sys_stdout, sys_stderr = sys.stdout, sys.stderr
                        sys.stdout = sys.stderr = output

                        input_counter = {'i': 0}
                        def input_mock(prompt=''):
                            print(prompt, end='')
                            if input_counter['i'] < len(mock_inputs):
                                val = mock_inputs[input_counter['i']]
                                input_counter['i'] += 1
                                print(val)
                                return val
                            raise EOFError("⚠ Not enough input provided.")

                        local_env = {'input_mock': input_mock}
                        try:
                            safe_code = code_input.replace("input(", "input_mock(")
                            start = time.time()
                            exec(safe_code, {}, local_env)
                            elapsed = time.time() - start
                            return output.getvalue(), None, local_env, elapsed
                        except Exception:
                            return output.getvalue(), traceback.format_exc(), local_env, None
                        finally:
                            sys.stdout = sys_stdout
                            sys.stderr = sys_stderr

                    def highlight_error(code, tb):
                        lines = code.strip().split('\n')
                        match = re.search(r'line (\d+)', tb)
                        error_line = int(match.group(1)) if match else None
                        highlighted = []
                        for i, line in enumerate(lines, 1):
                            prefix = f"{i:02d}: " if i == error_line else f"   {i:02d}: "
                            style = "color:red;background-color:#ffe6e6;" if i == error_line else ""
                            highlighted.append(f"<pre style='{style}'>{prefix}{line}</pre>")
                        return "\n".join(highlighted)

                    # Run and capture result
                    out, err, env, runtime = run_code(code_input, mock_inputs)

                    if err:
                        st.error("Runtime Error:")
                        st.code(err)
                        st.markdown("Error Line Highlight:")
                        st.markdown(highlight_error(code_input, err), unsafe_allow_html=True)
                    else:
                        st.success(f"Ran in {runtime:.4f}s")
                        st.code(out if out.strip() else "No output")

                    # Download
                    st.download_button("Download Result", out if not err else err, file_name="debug_output.txt")

                    # Variables
                    st.subheader("Variables")
                    st.json({k: v for k, v in env.items() if not k.startswith('__')})

                    # Complexity
                    st.subheader("Complexity")
                    st.code("\n".join(str(c) for c in cc_visit(code_input)))

                    st.subheader("Maintainability")
                    st.write(mi_visit(code_input, True))

                    st.subheader("Code Insight")
                    def explain(code):
                        if "for" in code or "while" in code: return " Uses loops"
                        if "def " in code: return "Function defined"
                        if "import " in code: return "Modules used"
                        return "General Python logic"
                    st.info(explain(code_input))



if st.session_state.show_debugger:
    show_code_debugger_sidebar()

