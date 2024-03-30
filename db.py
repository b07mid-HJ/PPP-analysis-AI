import json
import psycopg2
import streamlit_authenticator as stauth

conn=psycopg2.connect(
            database="Jade-Chatbot",
            user="postgres",
            password="admin",
            host="127.0.0.1",
            port=5432,
        )

curr = conn.cursor()
def select_user(user_id):
    curr.execute("SELECT * FROM chatusers WHERE id = %s;", (user_id,))
    user = curr.fetchone()
    return user

def insert_user(username, hashed_password):
    curr.execute("INSERT INTO chatusers (username, hashed_password) VALUES (%s, %s);", (username, hashed_password))
    conn.commit()

def fetch_all_users():
    curr.execute("SELECT * FROM chatusers;")
    users = curr.fetchall()
    return users

def insert_or_update_conversation(username, convo,doc):
    """
    Inserts a new conversation entry into the ChatHistory table if the user doesn't exist.
    If the user does exist, updates their conversation.
    """
    curr.execute("SELECT convo FROM ChatHistory WHERE username = %s AND doc = %s;", (username,doc))
    result = curr.fetchone()

    if result is None:
        # User doesn't exist, insert a new record
        conversation_json = json.dumps(convo)
        curr.execute("INSERT INTO ChatHistory (username, convo, doc) VALUES (%s, %s, %s);", (username, conversation_json,doc))
    else:
        # User exists, update their conversation
        conversation_json = json.dumps(convo)
        curr.execute("UPDATE ChatHistory SET convo = %s WHERE username = %s AND doc = %s;", (conversation_json, username,doc))

    conn.commit()

def fetch_conversations_by_user_document(username,doc):
    """
    Fetches all conversations for a specific user and document.
    """
    curr.execute("SELECT convo FROM ChatHistory WHERE username = %s AND doc = %s;", (username,doc))
    conversations = curr.fetchall()
    if not conversations:  # Check if the user exists
        return []
    else:
        conversations = (conversations[0])[0]
    return conversations

def fetch_all_docments():
    curr.execute("SELECT doc FROM documents;")
    documents = curr.fetchall()
    return documents
def clear_chathistory(username,doc):
    curr.execute("DELETE FROM ChatHistory WHERE username = %s AND doc = %s;", (username,doc))
    conn.commit()