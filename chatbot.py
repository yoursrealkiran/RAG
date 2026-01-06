import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangGraph and Langchain imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config

# Load environment variables from a .env file (if it exists)
load_dotenv()

# --- Configuration and Initialization ---

# Set Groq API Key from environment variable; raise clear error if missing
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize HuggingFace embeddings model for vector representation of text
def get_embeddings_model():
    """Returns a cached HuggingFaceEmbeddings instance using the model defined in config."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

embeddings = get_embeddings_model()

# Initialize Chroma vector store for semantic search over documents
def get_vector_store(embed_func):
    """
    Returns a Chroma vector store instance using the given embedding function.
    Stops the Streamlit app if the Chroma DB cannot be loaded.
    """
    try:
        return Chroma(persist_directory=config.CHROMA_PERSIST_DIRECTORY, embedding_function=embed_func)
    except Exception as e:
        st.error(f"Error loading ChromaDB. Make sure '{config.CHROMA_PERSIST_DIRECTORY}' exists and is populated. Error: {e}")
        st.stop() # Stop the Streamlit app if DB cannot be loaded

vectordb = get_vector_store(embeddings)

# Initialize the ChatGroq language model
def get_chat_model():
    """
    Returns a ChatGroq instance configured with:
    - model: the specific LLM model to use
    - temperature: low for consistent/deterministic answers
    - max_tokens: limit response length
    """
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0, 
        max_tokens=400
    )

model = get_chat_model()

# --- LangGraph Node Definition ---
def call_model(state: MessagesState):
    """
    Defines a LangGraph 'model' node that takes conversation state and calls the LLM.
    
    Args:
        state (MessagesState): Current conversation messages.
        
    Returns:
        dict: Contains updated 'messages' after LLM response.
    """
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
        "Answer all questions to the best of your ability."
    )
    # Prepend the system message to conversation history
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

# Build and compile the LangGraph workflow
def get_langgraph_app():
    """
    Constructs and compiles the LangGraph workflow.
    - Adds a 'model' node for handling user queries.
    - Connects START node to 'model' node.
    - Uses in-memory checkpointer to persist conversation history.
    
    Returns:
        Compiled LangGraph app ready for invocation.
    """
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Use a simple in-memory checkpointer for conversation state
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

app = get_langgraph_app()

# --- Streamlit UI Setup ---

st.set_page_config(page_title="QA", layout="centered")
st.title("Ask doubts regarding Numerical Methods and Computational Fluid Dynamics 📚❓")

# Initialize session state for messages and thread ID
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores all previous chat messages
if "thread_id" not in st.session_state:
    # Unique identifier for this chat session
    st.session_state.thread_id = "streamlit_chat_session"

# Display previous messages (user and assistant) in Streamlit chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Logic ---

if prompt := st.chat_input("Ask your question..."):
    # 1. Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking for an answer..."):
            try:
                # Retrieve top-3 most relevant documents from Chroma DB
                docs = vectordb.similarity_search_with_score(prompt, k=3)

                # Convert retrieved docs to a DataFrame for easier handling
                _docs = pd.DataFrame(
                    [(prompt, doc[0].page_content, doc[0].metadata.get('source'), doc[0].metadata.get('page'), doc[1]) for doc in docs],
                    columns=['query', 'paragraph', 'document', 'page_number', 'relevant_score']
                )
                # Concatenate context paragraphs for model input
                current_context = "\n\n".join(_docs['paragraph'])

                # Construct the HumanMessage with context and question
                current_turn_message = HumanMessage(content=f"Context: {current_context}\n\nQuestion: {prompt}")

                # Invoke the LangGraph workflow, appending this message to the session's conversation history
                result = app.invoke(
                    {"messages": [current_turn_message]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )

                # Extract the assistant's response from LangGraph result
                ai_response = result['messages'][-1].content

                # Get source document and top 3 page numbers for reference
                source_document = _docs['document'][0] if not _docs.empty and 'document' in _docs.columns else "N/A"
                top_three_page_numbers = _docs['page_number'].drop_duplicates().head(3).astype(str).tolist()
                page_numbers_str = ', '.join(top_three_page_numbers) if top_three_page_numbers else "N/A"

                # # Format response for display with Markdown
                final_response = f"{ai_response}\n\n**Source Document**: {source_document}\n**Reference Page Numbers**: {page_numbers_str}"
                
                st.markdown(final_response)

                # Append assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "I encountered an error. Please try again."})
