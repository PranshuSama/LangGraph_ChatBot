import streamlit as st
from langgraph_backend import chatbot, llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import uuid
import json
from pathlib import Path
from langgraph_database_backend import chatbot,retrieve_all_threads

# **************************************** utility functions *************************

TITLE_STORE_PATH = Path(__file__).with_name('thread_titles.json')

def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

def make_thread_title(text):
    cleaned_text = " ".join(text.strip().split())
    if not cleaned_text:
        return 'New Chat'
    if len(cleaned_text) > 40:
        return f"{cleaned_text[:40]}..."
    return cleaned_text

def make_thread_title_with_llm(text):
    cleaned_text = " ".join(text.strip().split())
    if not cleaned_text:
        return 'New Chat'

    try:
        title_response = llm.invoke([
            SystemMessage(content=(
                "You create concise chat titles. "
                "Return only one short title, max 6 words, no quotes, no punctuation at end."
            )),
            HumanMessage(content=f"Create a chat title for this user message: {cleaned_text}"),
        ])
        generated_title = " ".join(title_response.content.strip().split())
        if not generated_title:
            return make_thread_title(cleaned_text)
        if len(generated_title) > 40:
            generated_title = f"{generated_title[:40]}..."
        return generated_title
    except Exception:
        return make_thread_title(cleaned_text)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'], title='New Chat')
    st.session_state['message_history'] = []

def add_thread(thread_id, title=None):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    if thread_id not in st.session_state['thread_titles']:
        st.session_state['thread_titles'][thread_id] = title or 'New Chat'
        save_thread_titles(st.session_state['thread_titles'])

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def load_saved_thread_titles():
    if not TITLE_STORE_PATH.exists():
        return {}

    try:
        with TITLE_STORE_PATH.open('r', encoding='utf-8') as file:
            loaded_data = json.load(file)
        if isinstance(loaded_data, dict):
            return {str(thread_id): str(title) for thread_id, title in loaded_data.items()}
    except Exception:
        pass

    return {}

def save_thread_titles(title_map):
    try:
        with TITLE_STORE_PATH.open('w', encoding='utf-8') as file:
            json.dump(title_map, file, ensure_ascii=True, indent=2)
    except Exception:
        pass

def build_thread_title_map(thread_ids):
    title_map = {}

    for thread_id in thread_ids:
        messages = load_conversation(thread_id)
        first_user_message = next(
            (msg.content for msg in messages if isinstance(msg, HumanMessage) and msg.content),
            None,
        )
        title_map[thread_id] = make_thread_title(first_user_message) if first_user_message else 'New Chat'

    return title_map


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = load_saved_thread_titles()

    missing_titles = [
        thread_id
        for thread_id in st.session_state['chat_threads']
        if thread_id not in st.session_state['thread_titles']
    ]
    if missing_titles:
        st.session_state['thread_titles'].update(build_thread_title_map(missing_titles))

    save_thread_titles(st.session_state['thread_titles'])

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    button_title = st.session_state['thread_titles'].get(thread_id, 'New Chat')
    if st.sidebar.button(button_title, key=f"thread_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    current_thread_id = st.session_state['thread_id']
    if st.session_state['thread_titles'].get(current_thread_id, 'New Chat') == 'New Chat':
        st.session_state['thread_titles'][current_thread_id] = make_thread_title_with_llm(user_input)
        save_thread_titles(st.session_state['thread_titles'])

    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

     # first add the message to message_history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})