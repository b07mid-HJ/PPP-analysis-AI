import db

l=db.fetch_all_docments()
ll=[i[0] for i in l]
print(ll)