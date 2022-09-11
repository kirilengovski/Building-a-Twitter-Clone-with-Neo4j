from py2neo import Graph

graph = Graph('bolt://neo4j@localhost:7687', user="neo4j", password="123")

crypto = ["nebraskangooner", "Rager", "scottmelker", "cameron"]
finance = ["TMFJMo", "scottmelker", "morganhousel", "LizAnnSonders"]
food = ["epicurious", "SpoonUniversity", "Foodimentary", "Ottolenghi"]
health = ["WebMD", "robbwolf", "NutritionDiva", "DailyFitnessTip"]
tech = ["jeffweiner", "RyersonDMZ", "robbwolf", "TechCrunch"]
music = ["ElliottWilson", "gillespeterson", "MusicFactsFun", "future_of_music"]
christian = ["JoelOsteen", "timkellernyc", "biblegateway", "SheReadsTruth"]
muslim = ["YasminMogahed", "boonaamohammed", "muftimenk", "BonsaiSky"]
bhudhism = ["Buddhism_Now", "nobodhi", "chademeng", "dzigarkongtrul"]

all_cats = [crypto, finance, food, health, tech, music, christian, muslim, bhudhism]

tf_idf = """
    MATCH (u1:User {username: $username})<-[:BY]-(p1:Post)<-[:KEYWORD_IN]-(k:Keyword)-[:KEYWORD_IN]->(p2:Post)-[:BY]->(u2:User)
    WHERE u1.username <> u2.username
    AND NOT (u1)-[:FOLLOWS]->(u2)
    WITH u1, u2, COUNT(distinct k) as intersection, COLLECT(distinct k.name) as i
    MATCH (u1)<-[:BY]-(p3:Post)<-[:KEYWORD_IN]-(k1:Keyword)
    WITH u1, u2, intersection, i, COLLECT(distinct k1.name) as s1
    MATCH (u2)<-[:BY]-(p4:Post)<-[:KEYWORD_IN]-(k2:Keyword)
    WITH u1, u2, intersection, i, s1, COLLECT(distinct k2.name) as s2
    
    WITH u1.username as user1, u2.username as user2, intersection, i, s1, s2
    
    WITH user1, user2, intersection, i, s1 + [x in s2 WHERE NOT x IN s1] AS union, s1, s2
    
    RETURN user1, user2, s1, s2, i, ((1.0*intersection)/SIZE(union)) AS jaccard ORDER BY jaccard DESC LIMIT 3
    """

wordnet = """
    MATCH (u1:User {username: $username})<-[:BY]-(p1:Post)<-[:KEYWORD_IN]-(k:Keyword)-[:KEYWORD_IN]->(p2:Post)-[:BY]->(u2:User)
    WHERE u1.username <> u2.username
    OPTIONAL MATCH (u1)<-[:BY]-(p3:Post)<-[:KEYWORD_IN]-(k2:Keyword)-[:REFERS_TO]->(n:ontolex__LexicalEntry)-[:ontolex__sense]->()-[:ontolex__isLexicalizedSenseOf]->()<-[:ontolex__isLexicalizedSenseOf]-()<-[:ontolex__sense]-()<-[:REFERS_TO]-(k3:Keyword)-[:KEYWORD_IN]->(p4:Post)-[:BY]->(u2:User)
    WHERE u1.username <> u2.username
    WITH u1, u2, COUNT(distinct k) as intersection1, COUNT(distinct k2) as intersection2, COLLECT(distinct k3.name) as k3, COLLECT(distinct k.name) as i1, COLLECT (distinct k2.name) as i2
    MATCH (u1)<-[:BY]-(p3:Post)<-[:KEYWORD_IN]-(k1:Keyword)
    WITH u1, u2, intersection1+intersection2 as intersection, i2, k3, i1+i2 as i, COLLECT(distinct k1.name) as s1
    MATCH (u2)<-[:BY]-(p4:Post)<-[:KEYWORD_IN]-(k2:Keyword)
    WITH u1, u2, i2, k3, intersection, i, s1, COLLECT(distinct k2.name) as s2
    
    WITH u1.username as user1, i2, k3, u2.username as user2, intersection, i, s1, s2
    
    WITH user1, user2, intersection, i, i2, k3, s1 + [x in s2 WHERE NOT x IN s1] AS union, s1, s2
    
    RETURN user1, user2, s1, s2, i, i2, k3, intersection, size(i), ((1.0*intersection)/SIZE(union)) AS jaccard ORDER BY jaccard DESC LIMIT 3
    """

# print(all_cats)
def evaluate(model):
    tp, fp = 0, 0
    for i in range (0, 9):
        for j in range (0, 4):
			# print(all_cats[i][j])
            temp = []
            result = graph.run(model, username = all_cats[i][j]).data()
            for r in result:
                temp.append(r["user2"])
            for t in temp:
                if t in all_cats[i]:
                    tp += 1
                else:
                    fp += 1
    precision = tp / (tp + fp)
    print(tp, fp)
    return precision

tf_idf = evaluate(tf_idf)
wordnet = evaluate(wordnet)

print(tf_idf)
print(wordnet)






