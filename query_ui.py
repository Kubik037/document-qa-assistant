import json
from datetime import datetime

import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="Document QA Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
         height: 100%;
         margin: 0;
         padding: 0;
     }
    .main-header {
         font-size: 2.5rem;
         font-weight: bold;
         margin-bottom: 1rem;
         color: #1E3A8A;
    }
    .chat-message {
        background-color: #121212;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        position: relative;
    }
    .user-message {
        background-color: #121212;
        border-left: 5px solid #3B82F6;
    }
    .assistant-message {
        background-color: #121212;
        border-left: 5px solid #10B981;
    }
    .message-time {
        font-size: 0.7rem;
        color: #6B7280;
        position: absolute;
        right: 0.5rem;
        top: 0.5rem;
    }
    .source-block {
        background-color: #121212;
        color: #FFFFFF;
        padding: 0.5rem 0.5rem 0.5rem 1rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        font-family: monospace; 
        color: #4B5563;
    }
    /* Buttons */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
    }
    .clear-button > button {
        background-color: #DC2626;
    }
    .clear-button > button:hover {
        background-color: #B91C1C;
    }
    /* Loader dots animation */
    .thinking-dots {
        display: flex;
        align-items: center;
    }
    .dot {
        height: 10px;
        width: 10px;
        margin-right: 5px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 1.5s infinite ease-in-out;
    }
    .dot:nth-child(1) { animation-delay: 0s; }
    .dot:nth-child(2) { animation-delay: 0.3s; }
    .dot:nth-child(3) { animation-delay: 0.6s; }
    @keyframes pulse {
        0% { transform: scale(0.8); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.8); opacity: 0.5; }
    }
    .input-container {
        position: fixed !important;
        bottom: 0            !important;
        left:   0            !important;
        width:  100vw        !important;
        background-color: #121212;
        padding:          1rem;
        z-index:         1000;
        box-sizing:    border-box;
    }
    .chat-container {
        position: fixed;
        top:    6rem;   
        bottom: 6rem;     
        left:   1rem;      
        right:  1rem;
        overflow-y: auto;
        padding-bottom: 1rem;
        box-sizing: border-box;
    }
</style>
""", unsafe_allow_html=True)

# Session State init
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
if 'thinking' not in st.session_state:
    st.session_state.thinking = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "message_sent" not in st.session_state:
    st.session_state.message_sent = False
if "current_response" not in st.session_state:
    st.session_state.current_response = ""
if "sources" not in st.session_state:
    st.session_state.sources = []

with st.sidebar:
    st.markdown("### Document QA Settings")
    endpoint = st.radio(
        "Choose retrieval method:",
        options=["Classic QA", "Docs Retrieval only"],
        index=0,
        horizontal=True
    )
    st.markdown("### Conversation Options")
    include_history = st.checkbox("Include conversation history in queries", value=True)
    rerank = st.checkbox("Use reranking to improve source retrieval (takes longer)", value=True)
    st.markdown("### Knowledge Base Info")
    st.info(
        "This application searches through your document database to answer questions. "
        "Ask questions about your documents to get relevant answers with source citations."
    )
    st.markdown("### Conversation Management")
    if st.button("Clear History", key="clear_sidebar", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.rerun()

st.markdown("<h1 class='main-header'>📚 Document QA Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "Ask questions about your private document database. The system will retrieve relevant answers with source citations."
)

api_url = (
    "http://localhost:5000/ask_stream"
    if endpoint == "Classic QA"
    else "http://localhost:5000/source_retrieval"
)

def send_message():
    """Function to handle message sending and clear the input"""
    if st.session_state.user_input.strip() and not st.session_state.message_sent:
        user_message = st.session_state.user_input
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.session_state.message_sent = True
        st.session_state.user_input = ""
        st.session_state.thinking = True

# Chat area
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
if not st.session_state.chat_history:
    st.info("Start a conversation by asking a question below.")
else:
    for msg in st.session_state.chat_history:
        ts = msg.get("timestamp", "")
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-message user-message'>"
                f"<div class='message-time'>{ts}</div>"
                f"<strong>You:</strong> {msg['content']}"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='chat-message assistant-message'>"
                f"<div class='message-time'>{ts}</div>"
                f"<strong>Assistant:</strong> {msg['content']}"
                "</div>",
                unsafe_allow_html=True
            )
            if msg.get("sources"):
                st.markdown("<div class='source-block'><strong>Sources:</strong>", unsafe_allow_html=True)
                for s in msg["sources"]:
                    st.markdown(f"- {s}")
                st.markdown("</div>", unsafe_allow_html=True)

    # Thinking animation
    if st.session_state.thinking:
        st.markdown(
            "<div class='chat-message assistant-message'>"
            "<strong>Assistant:</strong>"
            "<div class='thinking-dots'>"
            "<span class='dot'></span><span class='dot'></span><span class='dot'></span>"
            "</div></div>",
            unsafe_allow_html=True
        )
st.markdown("</div>", unsafe_allow_html=True)

placeholder = st.empty()

st.markdown("<div class='input-container'>", unsafe_allow_html=True)
question = st.text_input(
    "Ask a question:",
    placeholder="E.g., How do I add a new language to the configuration translations?",
    key="user_input",
    on_change=send_message,
    help="Press Enter to send"
)
st.markdown("</div>", unsafe_allow_html=True)

# Handle API streaming when thinking
if st.session_state.thinking and st.session_state.chat_history:
    last_user = next(
        (m for m in reversed(st.session_state.chat_history) if m["role"] == "user"), None
    )
    if last_user:
        payload = {
            "question": last_user["content"],
            "context": [
                {"role": m["role"], "content": m["content"]}
                for m in (st.session_state.chat_history[:-1]
                          if include_history else [])
            ],
            "conversation_id": st.session_state.conversation_id,
            "rerank": rerank
        }
        full_response = ""
        sources = []
        try:
            if endpoint == "Classic QA":
                with requests.post(
                        api_url,
                        json=payload,
                        stream=True,
                        headers={"Accept": "text/event-stream"}
                ) as resp:
                    resp.raise_for_status()

                    st.session_state.thinking = False
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = line.decode("utf-8")
                        if not chunk.startswith("data: "):
                            continue

                        data = json.loads(chunk[6:])

                        # Append token and update the placeholder
                        if "token" in data:
                            full_response += data["token"]
                            placeholder.markdown(
                                f"<div class='chat-message assistant-message'>"
                                f"  <div class='message-time'>{datetime.now().strftime('%H:%M:%S')}</div>"
                                f"  <strong>Assistant:</strong> {full_response}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        # Collect sources if sent
                        if "sources" in data:
                            sources = data["sources"]

                        # Stop when the stream signals done
                        if data.get("done"):
                            break
            else:
                resp = requests.post(api_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                full_response = data.get("retrieved_chunks", "")
                sources = data.get("sources", [])

            msg = {
                "role": "assistant",
                "content": full_response if full_response else "No relevant information found",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "sources": sources if sources else []
            }
            st.session_state.chat_history.append(msg)
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {e}",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        finally:
            st.session_state.thinking = False
            st.session_state.message_sent = False
            st.rerun()

st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.8rem;'>"
    "Document QA Assistant • Powered by Local LLM"
    "</div>",
    unsafe_allow_html=True
)
