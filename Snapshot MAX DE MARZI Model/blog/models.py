from py2neo import Graph, Node, Relationship, NodeMatcher
from passlib.hash import bcrypt
from datetime import datetime
import uuid
import time
from datetime import timedelta, datetime
import tweepy
import csv


graph = Graph('bolt://neo4j@localhost:7687', user="neo4j", password="123")
matcher = NodeMatcher(graph)


class User:
    def __init__(self, username):
        self.username = username

    def find(self):
        user = graph.nodes.match("User", username = self.username).first()
        return user

    def register(self, password):
        if not self.find():
            user = Node("User", username=self.username, password=bcrypt.encrypt(password))
            graph.create(user)
            return True
        return False

    def verify_password(self, password):
        user = self.find()
        if not user:
            return False
        return bcrypt.verify(password, user["password"])

    def manual_post_add(self, text, k):
        user = self.find()
        post = Node(
            "Post",
            id=str(uuid.uuid4()),
            text=text,
            timestamp=int((datetime.now() - timedelta(days=k)).strftime("%s")),
            date=(datetime.today() - timedelta(days=k)).strftime("%F")
        )

        graph.create(Relationship(user, "PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.now() - timedelta(days=k)), post))
        graph.create(Relationship(post, ":BY", user))

        #
        # for mention in mentions:
        #     user2 = graph.nodes.match("User", username=mention).first()
        #     graph.create(Relationship(post, "MENTIONS", user2))

    def add_post(self, tags, text):
        user = self.find()
        post = Node(
            "Post",
            id=str(uuid.uuid4()),
            text=text,
            timestamp=int(datetime.now().strftime("%s")),
            date=datetime.now().strftime("%F")
        )

        graph.create(Relationship(user, "PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.now()), post))
        graph.create(Relationship(post, ":BY", user))

        tag_rel = Relationship.type("TAGGED")
        for tag in tags:
            t = Node("Tag", name=tag)
            t.__primarylabel__ = "Tag"
            t.__primarykey__ = "name"
            graph.merge(tag_rel(t, post))
        #
        # for mention in mentions:
        #     user2 = graph.nodes.match("User", username=mention).first()
        #     graph.create(Relationship(post, "MENTIONS", user2))

    def like_post(self, post_id):
        user = self.find()
        post = graph.nodes.match("Post", id = post_id).first()
        graph.create(Relationship(user, "LIKES", post))

    def recent_posts(self, n):
        query = """
        MATCH (user:User)-[r]->(p1:Post)
        WHERE user.username = $username
        AND TYPE(r) in $days
        OPTIONAL MATCH (p1)<-[:TAGGED]-(tag:Tag)
        RETURN p1 as post, COLLECT(distinct tag.name) AS tags, size((p1)<-[:LIKES]-()) as likes
        ORDER BY p1.date DESC, p1.timestamp DESC
        LIMIT $n
        """

        days = ["PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.now())]
        up_to = 30
        z = 1
        while z <= up_to:
            days.append("PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.today() - timedelta(days=z)))
            z += 1

        return graph.run(query, username=self.username, n=n, days=days)

    def similar_users(self, n):
        query = """
        MATCH (user1:User)<-[:BY]->(:Post)<-[:TAGGED]-(tag:Tag),
              (user2:User)<-[:BY]->(:Post)<-[:TAGGED]-(tag)
        WHERE user1.username = $username AND user1 <> user2
        WITH user2, COLLECT(DISTINCT tag.name) AS tags, COUNT(DISTINCT tag.name) AS tag_count
        ORDER BY tag_count DESC LIMIT $n
        RETURN user2.username AS similar_user, tags
        """
        return graph.run(query, username=self.username, n=n)

    def commonality_of_user(self, user):
        query1 = """
        MATCH (user1:User)<-[:BY]-(post:Post)<-[:LIKES]-(user2:User)
        WHERE user1.username = $username1 AND user2.username = $username2
        RETURN COUNT(post) AS likes
        """

        likes = graph.evaluate(query1, username1=self.username, username2=user.username)
        likes = 0 if not likes else likes

        query2 = """
        MATCH (user1:User)<-[:BY]-(:Post)<-[:TAGGED]-(tag:Tag),
              (user2:User)<-[:BY]->(:Post)<-[:TAGGED]-(tag)
        WHERE user1.username = $username1 AND user2.username = $username2
        RETURN COLLECT(DISTINCT tag.name) AS tags
        """

        tags = graph.evaluate(query2, username1=self.username, username2=user.username)

        return {"likes": likes, "tags": tags}

    def follow_user(self, user2):
        user1 = self.find()
        user2 = graph.nodes.match("User", username=user2.username).first()
        follow_rel = Relationship.type("FOLLOWS")
        graph.merge(follow_rel(user1, user2), "User", "username")

    def unfollow_user(self, user2):
        username1 = self.find()["username"]
        username2 = graph.nodes.match("User", username=user2.username).first()["username"]
        query = """
        MATCH (user1:User)-[r:FOLLOWS]->(user2:User)
        WHERE user1.username = $username1 AND user2.username = $username2
        RETURN r
        """
        rel = graph.evaluate(query, username1=username1, username2=username2)
        graph.separate(rel)
        return rel

    def is_following(self, username2):
        query = """
        MATCH (user1:User)-[:FOLLOWS]->(user2:User)
        WHERE user1.username = $username1 and user2.username = $username2
        RETURN user1, user2
        """

        return graph.evaluate(query, username1=self.username, username2=username2)

    def block_user(self, user2):
        user1 = self.find()
        user2 = graph.nodes.match("User", username=user2.username).first()
        block_rel = Relationship.type("BLOCKS")
        graph.merge(block_rel(user1, user2), "User", "username")
        query = """
        MATCH (user1:User)-[r:FOLLOWS]-(user2:User)
        WHERE user1.username = $username1 AND user2.username = $username2
        RETURN r
        """
        follow_rel = graph.evaluate(query, username1=self.username, username2=user2["username"])
        if follow_rel is not None:
            graph.separate(follow_rel)

    def unblock_user(self, user2):
        username1 = self.find()["username"]
        username2 = graph.nodes.match("User", username=user2.username).first()["username"]
        query = """
        MATCH (user1:User)-[r:BLOCKS]->(user2:User)
        WHERE user1.username = $username1 AND user2.username = $username2
        RETURN r
        """
        rel = graph.evaluate(query, username1=username1, username2=username2)
        graph.separate(rel)

    def is_blocked(self, username2):
        query = """
        MATCH (user1:User)-[:BLOCKS]-(user2:User)
        WHERE user1.username = $username1 and user2.username = $username2
        RETURN user1, user2
        """

        return graph.evaluate(query, username1=self.username, username2=username2)

    def get_accounts(self):
        query = """
        MATCH (u1:User)-[r]-(u2:User)
        WHERE u1.username = $username
        RETURN r, TYPE(r) as type
        """

        result = graph.run(query, username=self.username).data()
        related_users = {"blocks": [], "following": [], "followers": []}
        for r in result:
            if r["type"] == "BLOCKS" and r["r"].start_node["username"] == self.username:
                related_users["blocks"].append(r["r"].end_node["username"])
            elif r["type"] == "FOLLOWS" and r["r"].start_node["username"] == self.username:
                related_users["following"].append(r["r"].end_node["username"])
            elif r["type"] == "FOLLOWS" and r["r"].start_node != self.username:
                related_users["followers"].append(r["r"].start_node["username"])

        return related_users


# HOME PAGE NEWSFEED
def todays_recent_posts(n, username):
    query = """
    MATCH (u1:User)-[:FOLLOWS]->(u2:User)-[r]->(post:Post)
    WHERE u1.username = $username
    AND TYPE(r) in $days
    OPTIONAL MATCH (post)<-[:TAGGED]-(tag:Tag)
    RETURN post, u2.username as username, COLLECT(distinct tag.name) AS tags, size((post)<-[:LIKES]-()) as likes
    ORDER BY post.timestamp DESC
    LIMIT $n
    """
    days = ["PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.now())]
    up_to = 3
    z = 1
    while z <= up_to:
        days.append("PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.today() - timedelta(days=z)))
        z += 1
    return graph.run(query, n=n, username=username, days=days)


def add_follows():
    u_name = "ana"
    me = "kiril"
    user_me = graph.nodes.match("User", username=me).first()
    for u in range(0, 350):
        u_name = u_name + "a"
        user_other = graph.nodes.match("User", username=u_name).first()
        graph.create(Relationship(user_me, "FOLLOWS", user_other))


def profile_retrieval_experiment():

    accounts = []
    u_name = "ana"
    for u in range(0, 350):
        u_name = u_name + "a"
        accounts.append(u_name)

    accounts = accounts * 50

    query = """
        MATCH (user:User)-[r]->(p1:Post)
        WHERE user.username = $username
        AND TYPE(r) in $days
        OPTIONAL MATCH (p1)<-[:TAGGED]-(tag:Tag)
        RETURN p1 as post, COLLECT(distinct tag.name) AS tags, size((p1)<-[:LIKES]-()) as likes
        ORDER BY p1.timestamp DESC
        LIMIT 10
        """

    days = ["PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.now())]
    up_to = 5
    z = 1
    while z <= up_to:
        days.append("PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.today() - timedelta(days=z)))
        z += 1

    num_of_profiles_retrieved = []

    for x in range(0, 1000):
        start_time = time.time()
        seconds = 0.2
        counter = 0
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            graph.run(query, username=accounts[counter], days=days)

            if elapsed_time > seconds:
                break
            else:
                counter += 1

        num_of_profiles_retrieved.append(counter)

        print(counter)

    avg_num_of_profiles_retrieved = sum(num_of_profiles_retrieved) / len(num_of_profiles_retrieved)
    print("Total number of profiles retrieved in 1000 seconds" + str(sum(num_of_profiles_retrieved)))
    print("Average number of profiles retrieved per 1 second:" + str(avg_num_of_profiles_retrieved))

    # with open("dataset.csv") as fp:
    #     reader = csv.reader(fp, delimiter=",")
    #     next(reader, None)
    #     data_read = [row for row in reader]
    #
    # for x in data_read:
    #     if not x[0] in accounts:
    #         accounts.append(x[0])


# profile_retrieval_experiment()


# add_follows()

# create_dataset()
# populate_database()
# start = time.time()
# populate_database()
# end = time.time()
# total = end-start
# print(total)


# equivalent gregorian calendar
# def gregorian_calendar(graph, time1=None, node1=None):
#
#     if time1 is None:
#         time1 = datetime.now()
#
#     # gregorian calendar node
#     gregorian_node = Node("Calendar", calendar_type="Gregorian")
#     gregorian_node.__primarylabel__ = list(gregorian_node.labels)[0]
#     gregorian_node.__primarykey__ = "calendar_type"
#     graph.merge(gregorian_node)
#
#     # year node
#     that_year_node = Node("Year", year=time1.year, key=time1.strftime("%Y"))
#     that_year_node.__primarylabel__ = list(that_year_node.labels)[0]
#     that_year_node.__primarykey__ = "year"
#     graph.merge(that_year_node)
#
#     # calendar has year
#     rel = Relationship(gregorian_node, "YEAR", that_year_node)
#     graph.merge(rel)
#
#     # month node
#     that_month_node = Node("Month", month=time1.month, key=time1.strftime("%m-%Y"))
#     that_month_node.__primarylabel__ = list(that_month_node.labels)[0]
#     that_month_node.__primarykey__ = "month"
#     graph.merge(that_month_node)
#
#     # year has month
#     rel = Relationship(that_year_node, "MONTH", that_month_node)
#     graph.merge(rel)
#
#     # day node
#     that_day_node = Node("Day", day=time1.day, key=time1.strftime("%d-%m-%Y"))
#     that_day_node.__primarylabel__ = list(that_day_node.labels)[0]
#     that_day_node.__primarykey__ = "day"
#     graph.merge(that_day_node)
#
#     # month has day
#     rel = Relationship(that_month_node, "DAY", that_day_node)
#     graph.merge(rel)
#
#     # post was published on (gregorian) day
#     if node1 is not None:
#         rel = Relationship(node1, "ON", that_day_node)
#         graph.create(rel)
#
#
# def get_day_node(date):
#     gregorian_calendar(graph)
#     year = date.year
#     month = date.month
#     day = date.day
#     print(year, month, day)
#     query = """
#     MATCH (y:Year)-[:MONTH]->(m:Month)-[:DAY]-(d:Day)
#     WHERE d.day = $day AND m.month = $month AND y.year = $year
#     RETURN d
#     """
#     return graph.evaluate(query, day=day, month=month, year=year)
#
