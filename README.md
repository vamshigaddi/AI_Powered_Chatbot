# AI_Powered_Chatbot

This is an AI-powered chatbot designed to assist with customer support. The chatbot uses advanced language models to provide quick, helpful, and accurate responses to user queries. It is designed for businesses or platforms that need an automated solution for handling customer inquiries.

## Features

- **AI-Powered Responses:** Uses Groq and HuggingFace embeddings to power the chatbot's understanding and response generation.
- **Document-Based Knowledge:** The chatbot leverages stored documents (in `.txt` format) to provide context and relevant answers.
- **Vector Database Support:** Utilizes Chroma for storing and retrieving document embeddings.
- **Persistent Embeddings:** Embeddings are saved to disk, allowing for persistent data that is reused to avoid regeneration each time.
- **Query Handling:** The chatbot uses a templated prompt to generate concise, friendly, and informative responses based on user input.
  
## Prerequisites

Before running the project, ensure that you have the following installed:

- Python 3.10+
## Get Started
### Installing Dependencies

Clone the repository:

```bash
git clone https://github.com/vamshigaddi/AI_Powered_Chatbot.git
cd AI_Powered_Chatbot
```
create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # windows
```
Install the requirements
```bash
pip install -r requirements.txx
```
## Environment Setup
- Add your API key in the .env file:
- To get Groq api_key visit this website [Groq](https://groq.com/)

## Run the Application
```bash
fastapi dev app.py
```
- This will start the server locally on http://127.0.0.1:8000.
