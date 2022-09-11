**Snapshot of the Twitter Clone with my Max De Marzi's model after Week 4**

Make sure fresh Neo4J database is created and running first by opening the Neo4J applciation, creating a database by providing a name and a password and then starting that database. 

Then, navigate to the folder where the code is stored and:

pip install "all requirements listed bellow"

In models.py, change the graph to point to your local database and password:
graph = Graph('bolt://neo4j@localhost:7687', user="neo4j", password="your neo4j db password")

Then run the application from the same folder:
flask run

Go to:
http://127.0.0.1:5000/

Requirements: (Eg. pip install bcrypt)
- bcrypt
- cffi
- click
- Flask
- Jinja2
- MarkupSafe
- passlib
- py2neo
- pycparser
- Werkzeug

