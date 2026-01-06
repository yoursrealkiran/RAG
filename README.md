# QA Chat App for Numerical Methods and Computational Fluid Dynamics

This is an interactive web application for **context-aware question answering** over documents. It leverages **semantic search**, **LangGraph workflow**, and **ChatGroq LLM** to provide concise, source-backed answers to user queries.

---

## Features

- **Contextual Answers**: Retrieves the most relevant passages from your documents and generates answers using an LLM.
- **Conversation Memory**: Maintains chat history for follow-up questions using LangGraph's in-memory checkpointer.
- **Source Referencing**: Displays the source document and page numbers for transparency.
- **Easy-to-Use UI**: Built with Streamlit for a responsive chat experience.
- **Robust Error Handling**: Gracefully handles missing API keys, empty vector DBs, and other runtime errors.

---

## Demo

![Demo](images/Screenshot.png)

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yoursrealkiran/RAG.git
cd RAG
