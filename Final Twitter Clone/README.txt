**Snapshot of the Twitter Clone with my Solution model: FINAL**

This folder contains all the features I imnplemented throughout the project from basic Twitter functionalities to the Advnaced Search.

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

To use the Recommendation Engines, the TF-IDF algorithm function has to be called by uncommenting it from the botoom of models.py. Before this is done, a Neo4J database needs to be populated using the dataset provided in dataset.csv or creating posts through the user itnerface by starting the application and navigating to http://127.0.0.1:5000/. To import the Wordnet, use the code provided in the Appnedix C of the report. To donwload the Wordnet dataset go to https://github.com/globalwordnet/english-wordnet. 

To use Advanced Search functionality, similarly, the Google Cloud' entity extraction function has to be ran after the database is populated with the provided dataset or you can just create your own posts by using the user interfcae instead of loading the whole dataset. To import Wiki Categories as knoweldge grah, follow this: https://jbarrasa.com/2017/04/26/quickgraph6-building-the-wikipedia-knowledge-graph-in-neo4j-qg2-revisited/

As these are complex and depend on thrid party data, please contact me on: engovskiK@cardiff.ac.uk for more details or if you have any questions about starting the applications.
