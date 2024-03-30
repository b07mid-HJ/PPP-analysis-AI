import db

convo=[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi, how can I help you today?"}]
convo2=[{"role":"user","content":"hh"},{"role":"assistant","content":"Hi, how can I hhh you today?"}]
db.insert_or_update_conversation(username="B07MID",convo= convo)
db.insert_or_update_conversation(username="B07MID",convo= convo2)
print(db.fetch_conversations_by_user_document("B07MID"))