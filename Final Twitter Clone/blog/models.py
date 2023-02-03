from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from passlib.hash import bcrypt
import uuid
from datetime import timedelta, datetime
import re
import string
import tweepy
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv

lemmatizer = WordNetLemmatizer()


graph = Graph('bolt://neo4j@localhost:7687', user="neo4j", password="123")
matcher = NodeMatcher(graph)
rel_matcher = RelationshipMatcher(graph)


class User:
    def __init__(self, username):
        self.username = username

    def find(self):
        user = graph.nodes.match("User", username=self.username).first()
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

    def add_post(self, tags, text):
        user = self.find()
        post = Node(
            "Post",
            id=str(uuid.uuid4()),
            text=text,
            timestamp=int(datetime.now().strftime("%s")),
            date=datetime.now().strftime("%F")
        )

        query = """
        MATCH (u:User)-[r]->(p:Post)
        WHERE u.username = $username AND r.status = true
        RETURN r, TYPE(r) as type
        """

        result = graph.run(query, username=user["username"]).data()

        # If it is the first post ever for that user, create link between the user and a post
        if not result:
            rel = Relationship(user, "PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.now()), post, status=True)
            graph.create(rel)
            graph.create(Relationship(post, "BY", user))

        # Otherwise, chain the posts as a linked list with most recent one linked to the user
        else:
            rel = result[0]["r"]
            type_rel = result[0]["type"]
            old_post = rel.end_node
            graph.separate(rel)
            graph.create(Relationship(user, "PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.now()), post, status=True))
            graph.create(Relationship(post, "BY", user))
            graph.create(Relationship(post, type_rel, old_post))

        tag_rel = Relationship.type("TAGGED")
        for tag in tags:
            t = Node("Tag", name=tag)
            t.__primarylabel__ = "Tag"
            t.__primarykey__ = "name"
            graph.merge(tag_rel(t, post))

        # for mention in mentions:
        #     user2 = graph.nodes.match("User", username=mention).first()
        #     graph.create(Relationship(post, "MENTIONS", user2))

    def like_post(self, post_id):
        user = self.find()
        og_post = find_og(post_id)

        graph.create(Relationship(user, "LIKES", og_post))

    def recent_posts(self, n, viewer):
        query = """
        MATCH (user:User)-[r]->(p1:Post)
        WHERE user.username = $username
        AND (TYPE(r) STARTS WITH "RETWEETED_ON" OR TYPE(r) STARTS WITH "PUBLISHED_ON")
        MATCH (p1)-[f*0..6]->(p2:Post)
        WHERE all(x in f WHERE TYPE(x) STARTS WITH "RETWEETED_ON" OR TYPE(x) STARTS WITH "PUBLISHED_ON")
        WITH distinct [p1] as p1, COLLECT(distinct p2) as p2, user.username as publisher
        UNWIND p1 + p2 as post
        MATCH (post)-[:BY*1..2]-(u:User) 
        OPTIONAL MATCH (post)<-[:TAGGED]-(tag:Tag)
        OPTIONAL MATCH (post)-[:MENTIONS]->(u3:User)
        RETURN post.is_retweet as retweet, u.username as username, publisher, post, 
        COLLECT(distinct u3.username) as mentions, COLLECT(distinct tag.name) AS tags, 
        size((post)<-[:LIKES]-()) as likes
        ORDER BY post.timestamp DESC
        LIMIT $n
        """

        all_posts = graph.run(query, username=self.username, n=n).data()
        posts = []
        og_posts = []
        og_mentions = []
        og_tags = []
        for p in all_posts:
            if p["retweet"]:
                add_likes(p, p["post"]["id"])

            if viewer != self.username:
                if home_has_retweeted(viewer, p["username"], find_og(p["post"]["id"])["id"]):
                    og_posts.append(find_og(p["post"]["id"]))
                    p["post"]["temp"] = True
                else:
                    p["post"]["temp"] = False

            else:
                if self.has_retweeted(p["post"]["id"]):
                    og_posts.append(find_og(p["post"]["id"]))
                    p["post"]["temp"] = True
                else:
                    p["post"]["temp"] = False

        for p in og_posts:
            og_tags.append(get_tags(p["id"]))
            og_mentions.append(get_mentions(p["id"]))

        # Switcher for retweet / undo retweet
        for p in all_posts:
            if p["post"] in og_posts:
                p["post"]["temp"] = True

        # Loading tags and mentions from original post to retweet
        counter = 0
        for p in all_posts:
            if p["retweet"]:
                if len(og_posts) > 0:
                    p["tags"] = og_tags[counter]
                    p["mentions"] = og_mentions[counter]

                counter += 1
        return all_posts

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

    def advanced_search(self, search):
        username = self.find()["username"]

        depth_query = """
        MATCH
        path = (:Page)-[:IN_CATEGORY]->()-[: SUBCAT_OF * 0..]->(:RootCategory)
        RETURN
        max(length(path)) as maxTaxonomyDepth
        """
        depth = graph.evaluate(depth_query)

        ent_query = """
        MATCH (e:Entity)
        RETURN COLLECT (distinct e.ent)
        """
        entities = graph.evaluate(ent_query)

        similarity_scores = []

        query_og = """
          MATCH (u1:User)-[*]->(p1:Post)-[*1]->(fp1:Filtered_Post)-[:HAS_ENTITY]->(e1:Entity {ent: $search})-[:ABOUT]->(page1:Page)-[:IN_CATEGORY]->(c1:Category)
          WHERE u1.username = $username
          MATCH (u2:User)-[*]->(p2:Post)-[*1]->(fp2:Filtered_Post)-[:HAS_ENTITY]->(e2:Entity {ent: $e})-[:ABOUT]->(page2:Page)-[:IN_CATEGORY]->(c2:Category)
          WHERE NOT (u1)-[:FOLLOWS]-(u2)
          AND u1.username <> u2.username
          WITH  distinct u1, e1, c1, e2, c2, u2
          MATCH p = shortestPath((c1)-[:SUBCAT_OF*0..]-(c2))
          WITH u1, e1, c1, e2, c2, u2, length(p) + 2 as pathLen        
          RETURN [u2.username, e2.ent, pathLen, 1.0/(1+pathLen)] as similarity
          ORDER BY pathLen LIMIT 1
        """

        for e in entities:
            similarity_scores.append(graph.evaluate(query_og, username=username, search=search, e=e, taxonomyDepth=depth))

        similarity_scores = [i for i in similarity_scores if i]

        # Sum up the total similarity score for each user as one could have more than 1 common entity

        pre_results = {}
        for x in similarity_scores:
            if x[0] not in pre_results:
                pre_results[x[0]] = x[2]
            else:
                pre_results[x[0]] += x[2]

        pre_results = dict(sorted(pre_results.items(), key=lambda item: -item[1]))
        print(pre_results)
        results = list(pre_results.keys())[0:3]

        return results

    def get_users(self, search):
        username = self.find()["username"]
        query = """
        MATCH (u:User), (u2:User)
        WHERE u.username = $username AND u.username <> u2.username AND NOT (u)-[:BLOCKS]-(u2)
        RETURN COLLECT(distinct u2.username) as usernames
        """
        usernames = graph.evaluate(query, username=username)
        results = []
        for u in usernames:
            if search == u:
                results.append(u)

        return results

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

    def retweet(self, post_id):
        user = self.find()

        og_post = find_og(post_id)

        post = Node("Post",
                    id=str(uuid.uuid4()),
                    text=og_post["text"],
                    date=datetime.now().strftime("%F"),
                    timestamp=int(datetime.now().strftime("%s")),
                    is_retweet=True
                    )

        graph.create(Relationship(post, "BY", og_post))
        graph.create(Relationship(user, "RETWEETS", post))

        query = """
                MATCH (u:User)-[r]->(p:Post)
                WHERE u.username = $username AND r.status = true
                RETURN r, TYPE(r) as type
                """

        result = graph.run(query, username=user["username"]).data()

        # If it is the first post ever for that user, create link between the user and a post
        if not result:
            rel = Relationship(user, "RETWEETED_ON_" + "{:%Y_%m_%d}".format(datetime.now()), post, status=True)
            graph.create(rel)

        # Otherwise, chain the posts as a linked list with most recent one linked to the user
        else:
            rel = result[0]["r"]
            type_rel = result[0]["type"]
            old_post = rel.end_node
            graph.separate(rel)
            graph.create(Relationship(user, "RETWEETED_ON_" + "{:%Y_%m_%d}".format(datetime.now()), post, status=True))
            graph.create(Relationship(post, type_rel, old_post))

    def has_retweeted(self, post_id):
        user = self.find()
        query = """
        MATCH (u:User)-[:RETWEETS]->(p:Post)
        WHERE u.username = $username AND p.id = $post_id
        RETURN p.id
        """
        return graph.evaluate(query, username=user["username"], post_id=post_id)

    # TO DO

    def undo_retweet(self, post_id):
        user = self.find()
        query = """
        MATCH (u:User)-[r1]->(p1:Post)
        WHERE TYPE(r1) STARTS WITH "RETWEET" AND u.username = $username AND p1.id = $post_id
        OPTIONAL MATCH (p1)<-[r2]-(p2:Post)
        WHERE TYPE(r2) STARTS WITH "RETWEET"
        OPTIONAL MATCH (p1)<-[r4]-(u)
        WHERE TYPE(r4) STARTS WITH "RETWEETED_ON_"
        OPTIONAL MATCH (p1)-[r3]->(p3:Post)
        WHERE TYPE(r3) STARTS WITH "RETWEETED_ON_" OR TYPE(r3) STARTS WITH "PUBLISHED_ON_"
        WITH u as user, p1 as retweet_post, p2 as previous, p3 as og_post, COLLECT(distinct TYPE(r2)) as prev_to_p, 
        TYPE(r4) as u_to_p, COLLECT(distinct TYPE(r3)) as p_to_p
        RETURN user, retweet_post, previous, og_post, u_to_p, p_to_p, prev_to_p
        """

        is_retweet = graph.nodes.match("Post", id=post_id).first()["is_retweet"]
        if not is_retweet:
            post_id = find_retweet(post_id, user["username"])["id"]

        is_direct_rtw = is_direct_retweet(post_id, user["username"])
        if not is_direct_rtw:
            post_id = find_direct_retweet(post_id, user["username"])["id"]

        result = graph.run(query, username=user["username"], post_id=post_id).data()[0]
        og_rel = False
        # If there is a another post attached to the retweet on the linked list, re-attachment is done.
        if result["p_to_p"]:
            og_rel = result["p_to_p"][0]

        graph.delete(result["retweet_post"])

        if og_rel:
            if result["u_to_p"]:
                graph.create(Relationship(result["user"], og_rel, result["og_post"], status=True))
            else:
                graph.create(Relationship(result["previous"], og_rel, result["og_post"]))

    def get_recommended_users(self):
        result = jaccard_index(self.username)
        return result


def get_tags(post_id):
    query = """
    MATCH (p1:Post)-[:TAGGED]-(t:Tag)
    WHERE p1.id = $post_id
    RETURN COLLECT(distinct t.name) as tags
    """
    return graph.evaluate(query, post_id=post_id)


def get_mentions(post_id):
    query = """
    MATCH (p1:Post)-[:MENTIONS]-(u:User)
    WHERE p1.id = $post_id
    RETURN COLLECT(distinct u.username) as mentions
    """
    return graph.evaluate(query, post_id=post_id)


def find_og(post_id):
    query = """
    MATCH (p1:Post)-[:BY]->(p2:Post)
    WHERE p1.id = $post_id
    RETURN p2
    """

    og_post = graph.evaluate(query, post_id=post_id)

    if not og_post:
        og_post = graph.nodes.match("Post", id=post_id).first()

    return og_post


def find_retweet(post_id, username):
    query = """
    MATCH (p1:Post)<-[:BY]-(p2:Post)<-[:RETWEETS]-(u:User)
    WHERE p1.id = $post_id AND u.username = $username
    RETURN p2
    """

    # OPTIONAL MATCH (p1)-[:BY]->(p3:Post)<-[:BY]-(p2)<-[:RETWEETS]-(u:User)
    return graph.evaluate(query, post_id=post_id, username=username)


def is_direct_retweet(post_id, username):
    query = """
    MATCH (p1:Post)<-[:RETWEETS]-(u1:User)
    WHERE p1.id = $post_id AND u1.username = $username
    RETURN p1
    """
    return graph.evaluate(query, post_id=post_id, username=username)


def find_direct_retweet(post_id, username):
    query = """
    MATCH (p1:Post)-[:BY]->(p2:Post)<-[:BY]-(p3:Post)<-[:RETWEETS]-(u:User)
    WHERE p1.id = $post_id AND u.username = $username
    RETURN p3
    """

    return graph.evaluate(query, post_id=post_id, username=username)


def tag_search(name, n, username):
    user = User(username)
    user = user.find()
    query = """
    MATCH (u:User)<-[:BY]-(post:Post)<-[:TAGGED]-(tag:Tag)
    WHERE tag.name = $name
    OPTIONAL MATCH (post)-[:MENTIONS]->(u2:User)
    RETURN post.is_retweet as retweet, u.username as username, post, COLLECT(distinct u2.username) as mentions, 
    COLLECT(distinct tag.name) as tags, size((post)<-[:LIKES]-()) as likes
    ORDER BY post.timestamp DESC
    LIMIT $n
    """

    all_posts = graph.run(query, name=name, n=n).data()
    for p in all_posts:
        if home_has_retweeted(user["username"], p["username"], find_og(p["post"]["id"])["id"]):
            p["post"]["temp"] = True
        else:
            p["post"]["temp"] = False

    return all_posts


def get_likes(post_id):
    query = """
    MATCH (p:Post)
    WHERE p.id = $post_id
    RETURN size((p)<-[:LIKES]-())
    """
    return graph.evaluate(query, post_id=post_id)


def add_likes(retweet, post_id):
    og_post = find_og(post_id)
    og_post_likes = get_likes(og_post["id"])
    retweet["likes"] = og_post_likes


def home_has_retweeted(username1, username2, post_id):
    query = """
    MATCH (u1:User)-[:RETWEETS]->(p1:Post)-[:BY]->(p2:Post)-[:BY]->(u2:User)
    WHERE u1.username = $username1 AND u2.username = $username2 AND p2.id = $post_id
    WITH COLLECT(distinct p2.id) as retweets
    RETURN retweets
    """

    return graph.evaluate(query, username1=username1, username2=username2, post_id=post_id)


# HOME PAGE NEWSFEED
def todays_recent_posts(n, username):
    query = """
    MATCH (u1:User)-[:FOLLOWS]->(u2:User)-[r]->(p1:Post)
    WHERE u1.username = $username
    AND TYPE(r) in [$today, $yesterday, $today_r, $yesterday_r]
    AND NOT (u1)-[:BLOCKS]-(u2)
    OPTIONAL MATCH (p1)-[f*0..3]->(p2:Post)
    WHERE all(x in f WHERE TYPE(x) in [$today, $yesterday, $today_r, $yesterday_r])
    WITH distinct [p1] as p1, COLLECT(distinct p2) as p2, u2.username as publisher
    UNWIND p1 + p2 as post
    MATCH (post)-[:BY*1..2]->(u:User)
    OPTIONAL MATCH (post)<-[:TAGGED]-(tag:Tag)
    OPTIONAL MATCH (post)-[:MENTIONS]->(u3:User)
    RETURN post.is_retweet as retweet, u.username as username, publisher, COLLECT(distinct u3.username) as mentions, 
    post, COLLECT(distinct tag.name) as tags, size((post)<-[:LIKES]-()) as likes
    ORDER BY post.timestamp DESC
    LIMIT $n
    """

    today = "PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.now())
    today_r = "RETWEETED_ON_" + "{:%Y_%m_%d}".format(datetime.now())
    yesterday = "PUBLISHED_ON_" + "{:%Y_%m_%d}".format(datetime.today() - timedelta(days=1))
    yesterday_r = "RETWEETED_ON_" + "{:%Y_%m_%d}".format(datetime.today() - timedelta(days=1))

    all_posts = graph.run(query, n=n, username=username, today=today, yesterday=yesterday, today_r=today_r,
                          yesterday_r=yesterday_r).data()

    user = User(username)
    user = user.find()
    og_posts = []

    for p in all_posts:
        # Update likes to match the original post likes of a retweet
        if p["retweet"]:
            add_likes(p, p["post"]["id"])
            og_posts.append(find_og(p["post"]["id"]))

        # Find retweets and display retweet or undo retweet buttons
        if home_has_retweeted(user["username"], p["publisher"], find_og(p["post"]["id"])["id"]):
            p["post"]["temp"] = True
        else:
            p["post"]["temp"] = False

    og_tags = []
    og_mentions = []

    for x in og_posts:
        og_tags.append(get_tags(x["id"]))
        og_mentions.append(get_mentions(x["id"]))

    for x in all_posts:
        if x["post"] in og_posts:
            x["post"]["temp"] = True

    counter = 0
    for x in all_posts:
        if x["retweet"]:
            x["tags"] = og_tags[counter]
            x["mentions"] = og_mentions[counter]
            counter += 1

    return all_posts


def get_all_posts():
    query = """
    MATCH (u:User)<-[:BY]-(post)
    WITH COLLECT(distinct post) as posts
    UNWIND posts as ps
    RETURN distinct ps.text as tweet, ps.id as id
    """

    return graph.run(query).data()


def get_some_posts():
    query = """
        MATCH (u:User)<-[:BY]-(post)
        WHERE u.username = "Eminem" OR u.username = "ra" OR u.username = "sa"
        WITH COLLECT(distinct post) as posts
        UNWIND posts as ps
        RETURN distinct ps.text as tweet, ps.id as id
        """
    return graph.run(query).data()


# NLP to find keywords from tweets using TF-IDF
def get_keywords(all):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')

    def clean_text(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Removes @mentions
        text = re.sub(r'#', '', text)  # Removes # symbol
        text = re.sub(r'RT[\s]+', '', text)  # Removing RT
        text = re.sub(r'https?:\/\/\S+', '', text)  # Removing links

        return text

    # https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    def demojify(text):
        regrex_pattern = re.compile(pattern="["
                                            u"\U0001F600-\U0001F64F"  # emoticons
                                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                            "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', text)

    posts = []
    ids = []
    string.punctuation = string.punctuation + '’'
    string.punctuation = string.punctuation + '…'
    string.punctuation = string.punctuation + '—'

    for i in all:
        posts.append(demojify(clean_text(i['tweet'])))

    for i in all:
        ids.append(i['id'])

    bag_of_words = {}
    total = len(posts)

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('wordnet')

    for i in range(0, total):
        split_post = posts[i].split(' ')
        split_post = [lemmatizer.lemmatize(x.translate(str.maketrans('', '', string.punctuation))) for x in split_post if x.lower().islower()
                      and lemmatizer.lemmatize(x.lower()) not in stopwords.words('english')]

        # Removing everything that doesn't contain characters including numbers
        # for w in split_post:
        #     if not w.islower():
        #         split_post.remove(w)
        bag_of_words[ids[i]] = split_post

    unique_words = []
    for k, v in bag_of_words.items():
        for x in v:
            unique_words.append(x)

    num_of_words = {}
    for i in range(0, total):
        num_of_words[ids[i]] = dict.fromkeys(unique_words, 0)

    for k, v in num_of_words.items():
        for word in bag_of_words[k]:
            v[word] += 1

    def compute_tf(word_dict, bag_of_words):
        tf_dict = {}
        bag_of_words_count = len(bag_of_words)
        for word, count in word_dict.items():
            if bag_of_words_count > 0:
                tf_dict[word] = count / float(bag_of_words_count)
        return tf_dict

    tfs = {}
    for x in range(0, total):
        tfs[ids[x]] = compute_tf(num_of_words[ids[x]], bag_of_words[ids[x]])

    def compute_idf(documents):
        import math
        N = len(documents)
        idfDict = dict.fromkeys(documents[0].keys(), 0)
        for document in documents:
            for word, val in document.items():
                if val > 0:
                    idfDict[word] += 1

        for word, val in idfDict.items():
            idfDict[word] = math.log(N / float(val))
        return idfDict

    num_of_words_list = []
    for k, v in num_of_words.items():
        num_of_words_list.append(v)

    idfs = compute_idf(num_of_words_list)

    def compute_tf_idf(tfBagOfWords, idfs):
        tfidf = {}
        for word, val in tfBagOfWords.items():
            tfidf[word] = val * idfs[word]
        return tfidf

    tfidfs = {}
    for x in range(0, total):
        tfidfs[ids[x]] = compute_tf_idf(tfs[ids[x]], idfs)

    # Sorting the dictionaries and getting the top two keywords of each post
    top_twos = {}
    for k, v in tfidfs.items():
        v = dict(sorted(v.items(), key=lambda item: item[1], reverse=True))
        tfidfs[k] = v
        top_twos[k] = {k: v[k] for k in list(v)[:4]}

    # Creating keywords nodes and relationship to their posts
    for k, v in top_twos.items():
        post = graph.nodes.match("Post", id=k).first()
        for x, y in v.items():
            z = graph.nodes.match("Keyword", name=x.lower()).first()
            if not z:
                z = Node("Keyword", name=x.lower())
                graph.create(z)
            graph.create(Relationship(z, "KEYWORD_IN", post, TFIDF=y))
    print(top_twos)

def link_keywords_to_wordnet():
    query = """
    MATCH (k:Keyword)-[]-()-[:BY]-(u:User)
    RETURN COLLECT(k.name)
    """

    keywords = graph.evaluate(query)

    for key in keywords:
        a = graph.nodes.match("Keyword", name=key).first()
        b = graph.nodes.match("Resource", ontolex__canonicalForm=key).first()
        if b:
            graph.create(Relationship(a, "REFERS_TO", b))


def set_weights():
    query1 = """
    MATCH (k:Keyword)-[r:REFERS_TO]-(n:ontolex__LexicalEntry)-[r1:ontolex__sense]->()-[r2:ontolex__isLexicalizedSenseOf]->(def:ontolex__LexicalConcept)
    RETURN k.name as keyword, n.ontolex__canonicalForm as word, COUNT(distinct def) as num_of_definitions
    """

    query2 = """
    MATCH (n:ontolex__LexicalEntry)-[r1:ontolex__sense]->()-[r2:ontolex__isLexicalizedSenseOf]->(def:ontolex__LexicalConcept)<-[f:wn__hypernym*1]-(def2:ontolex__LexicalConcept)
    RETURN def.wn__definition as definition, COUNT(distinct def2) as num_of_hypernyms
    """

    query4 = """
    MATCH (n:ontolex__LexicalConcept)
    WHERE NOT EXISTS(n.num_of_hypernyms)
    RETURN n.wn__definition as definition
    """

    # results1 = graph.run(query1).data()
    # results2 = graph.run(query2).data()
    #
    # for x in results1:
    #     keyword = graph.nodes.match("Keyword", name=x.get("keyword")).first()
    #     word = graph.nodes.match("ontolex__LexicalEntry", ontolex__canonicalForm=x.get("word")).first()
    #     num_of_definitions = x.get("num_of_definitions")
    #     rel = rel_matcher.match([keyword, word], "REFERS_TO").first()
    #     rel["num_of_definitions"] = num_of_definitions
    #     graph.push(rel)
    # #
    # for x in results2:
    #     definition = graph.nodes.match("ontolex__LexicalConcept", wn__definition=x.get("definition")).first()
    #     definition["num_of_hypernyms"] = x.get("num_of_hypernyms")
    #     graph.push(definition)

    results4 = graph.run(query4).data()

    for x in results4:
        definition = graph.nodes.match("ontolex__LexicalConcept", wn__definition=x.get("definition")).first()
        definition["num_of_hypernyms"] = 0
        graph.push(definition)


# Calculate similarity between users using Jaccard Index
def jaccard_index(username):
    query = """
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
    
    RETURN user1, user2, s1, s2, i, ((1.0*intersection)/SIZE(union)) AS jaccard ORDER BY jaccard DESC LIMIT 5
    """

    query_2 = """
    MATCH (u1:User {username: $username})<-[:BY]-(p1:Post)<-[:KEYWORD_IN]-(k:Keyword)-[:KEYWORD_IN]->(p2:Post)-[:BY]->(u2:User)
    WHERE u1.username <> u2.username
    AND NOT (u1)-[:FOLLOWS]->(u2)
    OPTIONAL MATCH (u1)<-[:BY]-(p3:Post)<-[:KEYWORD_IN]-(k2:Keyword)-[:REFERS_TO]->(n:ontolex__LexicalEntry)-[:ontolex__sense]->()-[:ontolex__isLexicalizedSenseOf]->()<-[:ontolex__isLexicalizedSenseOf]-()<-[:ontolex__sense]-()<-[:REFERS_TO]-(k3:Keyword)-[:KEYWORD_IN]->(p4:Post)-[:BY]->(u2:User)
    WHERE u1.username <> u2.username
    AND NOT (u1)-[:FOLLOWS]->(u2)
    WITH u1, u2, COUNT(distinct k) as intersection1, COUNT(distinct k2) as intersection2, COLLECT(distinct k3.name) as k3, COLLECT(distinct k.name) as i1, COLLECT (distinct k2.name) as i2
    MATCH (u1)<-[:BY]-(p3:Post)<-[:KEYWORD_IN]-(k1:Keyword)
    WITH u1, u2, intersection1+intersection2 as intersection, i2, k3, i1+i2 as i, COLLECT(distinct k1.name) as s1
    MATCH (u2)<-[:BY]-(p4:Post)<-[:KEYWORD_IN]-(k2:Keyword)
    WITH u1, u2, i2, k3, intersection, i, s1, COLLECT(distinct k2.name) as s2
    
    WITH u1.username as user1, i2, k3, u2.username as user2, intersection, i, s1, s2
    
    WITH user1, user2, intersection, i, i2, k3, s1 + [x in s2 WHERE NOT x IN s1] AS union, s1, s2
    
    RETURN user1, user2, s1, s2, i, i2, k3, intersection, size(i), ((1.0*intersection)/SIZE(union)) AS jaccard ORDER BY jaccard DESC LIMIT 5
    """

    query_3 = """
    MATCH (u1:User {username: "BillGates"})<-[:BY]-(p1:Post)<-[:KEYWORD_IN]-(k:Keyword)-[:KEYWORD_IN]->(p2:Post)-[:BY]->(u2:User)
    WHERE u1.username <> u2.username
    OPTIONAL MATCH (u1)<-[:BY]-(p3:Post)<-[:KEYWORD_IN]-(k2:Keyword)-[r:REFERS_TO]->(n:ontolex__LexicalEntry)-[:ontolex__sense]->()-[:ontolex__isLexicalizedSenseOf]->(def:ontolex__LexicalConcept)<-[:ontolex__isLexicalizedSenseOf]-()<-[:ontolex__sense]-()<-[:REFERS_TO]-(k3:Keyword)-[:KEYWORD_IN]->(p4:Post)-[:BY]->(u2:User)
    WHERE u1.username <> u2.username
    WITH u1, u2, sum(1/r.num_of_definitions) as intersection1, sum(1 / (def.num_of_hypernyms_REAL+1)) as intersection2, COLLECT(distinct k3.name) as k3, COLLECT(distinct k.name) as i1, COLLECT (distinct k2.name) as i2
    MATCH (u1)<-[:BY]-(p3:Post)<-[:KEYWORD_IN]-(k1:Keyword)
    WITH u1, u2, intersection1+intersection2 as intersection, i2, k3, i1+i2 as i, COLLECT(distinct k1.name) as s1
    MATCH (u2)<-[:BY]-(p4:Post)<-[:KEYWORD_IN]-(k2:Keyword)
    WITH u1, u2, i2, k3, intersection, i, s1, COLLECT(distinct k2.name) as s2
    
    WITH u1.username as user1, i2, k3, u2.username as user2, intersection, i, s1, s2
    
    WITH user1, user2, intersection, i, i2, k3, s1 + [x in s2 WHERE NOT x IN s1] AS union, s1, s2
    
    RETURN user1, user2, s1, s2, i, i2, k3, intersection, size(i), ((1.0*intersection)/SIZE(union)) AS jaccard ORDER BY jaccard DESC LIMIT 5
    """

    return graph.run(query, username=username).data()


# Get tweets from Twitter dev account and populate the database
def create_dataset():
    log = ["Y3bFn6fbxhNMUrShsNRvqe50p", "CbF14gRI8VlkqCPdVHXp0chadDho8PRplN227OMtQHg4UciN3N",
           "3480572415-5mXRtGF9G36FcOnKHOz2o8pFaA3hATrUC51CCiN", "aYDBQBc9Zek49fbPJhlBbmh9T8Bz0BPelgicrEjdKEK0J"]

    consumer_key = log[0]
    consumer_secret = log[1]
    access_token = log[2]
    access_token_secret = log[3]

    authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)
    authenticate.set_access_token(access_token, access_token_secret)

    api = tweepy.API(authenticate, wait_on_rate_limit=True)

    tweets = {
        # 200
        "BillGates": api.user_timeline(screen_name="BillGates", count=300, lang="en", tweet_mode="extended"),
        "joerogan": api.user_timeline(screen_name="joerogan", count=300, lang="en", tweet_mode="extended"),
        "Genius": api.user_timeline(screen_name="Genius", count=300, lang="en", tweet_mode="extended"),
        "rihanna": api.user_timeline(screen_name="rihanna", count=300, lang="en", tweet_mode="extended"),
        "Eminem": api.user_timeline(screen_name="Eminem", count=300, lang="en", tweet_mode="extended"),
        "BarackObama": api.user_timeline(screen_name="BarackObama", count=300, lang="en", tweet_mode="extended"),
        # 5
        "TheEconomist": api.user_timeline(screen_name="TheEconomist", count=5, lang="en", tweet_mode="extended"),
        "World_Wildlife": api.user_timeline(screen_name="World_Wildlife", count=5, lang="en", tweet_mode="extended"),
        "POTUS": api.user_timeline(screen_name="POTUS", count=5, lang="en", tweet_mode="extended"),
        "DonaldJTrumpJr": api.user_timeline(screen_name="DonaldJTrumpJr", count=5, lang="en", tweet_mode="extended"),
        "Drake": api.user_timeline(screen_name="Drake", count=5, lang="en", tweet_mode="extended"),
        "CNN": api.user_timeline(screen_name="CNN", count=5, lang="en", tweet_mode="extended"),
        # 10
        "nytimes": api.user_timeline(screen_name="nytimes", count=10, lang="en", tweet_mode="extended"),
        "NASA": api.user_timeline(screen_name="NASA", count=10, lang="en", tweet_mode="extended"),
        "GordonRamsay": api.user_timeline(screen_name="GordonRamsay", count=10, lang="en", tweet_mode="extended"),
        "SkyFootball": api.user_timeline(screen_name="SkyFootball", count=10, lang="en", tweet_mode="extended"),
        "MoveTheWorldUK": api.user_timeline(screen_name="MoveTheWorldUK", count=10, lang="en", tweet_mode="extended"),
        "emmetlsavage": api.user_timeline(screen_name="emmetlsavage", count=10, lang="en", tweet_mode="extended"),
        # 50
        "NBA": api.user_timeline(screen_name="NBA", count=50, lang="en", tweet_mode="extended"),
        "oceana": api.user_timeline(screen_name="oceana", count=50, lang="en", tweet_mode="extended"),
        "DeepLifeQuotes": api.user_timeline(screen_name="DeepLifeQuotes", count=50, lang="en", tweet_mode="extended"),
        "pourmecoffee": api.user_timeline(screen_name="pourmecoffee", count=50, lang="en", tweet_mode="extended"),
        "justinbieber": api.user_timeline(screen_name="justinbieber", count=50, lang="en", tweet_mode="extended"),
        "taylorswift13": api.user_timeline(screen_name="taylorswift13", count=50, lang="en", tweet_mode="extended"),
        "TheEllenShow": api.user_timeline(screen_name="TheEllenShow", count=50, lang="en", tweet_mode="extended"),
        "Oprah": api.user_timeline(screen_name="Oprah", count=50, lang="en", tweet_mode="extended"),
        "espn": api.user_timeline(screen_name="espn", count=50, lang="en", tweet_mode="extended"),
        "KevinHart4real": api.user_timeline(screen_name="KevinHart4real", count=50, lang="en", tweet_mode="extended"),
        "ActuallyNPH": api.user_timeline(screen_name="ActuallyNPH", count=50, lang="en", tweet_mode="extended"),
        "Google": api.user_timeline(screen_name="Google", count=50, lang="en", tweet_mode="extended"),
        "kourtneykardash": api.user_timeline(screen_name="kourtneykardash", count=50, lang="en", tweet_mode="extended"),
        "NatGeo": api.user_timeline(screen_name="NatGeo", count=50, lang="en", tweet_mode="extended"),
        "elonmusk": api.user_timeline(screen_name="elonmusk", count=50, lang="en", tweet_mode="extended"),
        "jtimberlake": api.user_timeline(screen_name="jtimberlake", count=50, lang="en", tweet_mode="extended"),
        # Finance
        "LizAnnSonders": api.user_timeline(screen_name="LizAnnSonders", count=50, lang="en", tweet_mode="extended"),
        "morganhousel": api.user_timeline(screen_name="morganhousel", count=50, lang="en", tweet_mode="extended"),
        "scottmelker": api.user_timeline(screen_name="scottmelker", count=50, lang="en", tweet_mode="extended"),
        "TMFJMo": api.user_timeline(screen_name="TMFJMo", count=50, lang="en", tweet_mode="extended"),
        # Crypto
        "nebraskangooner": api.user_timeline(screen_name="nebraskangooner", count=50, lang="en", tweet_mode="extended"),
        "Rager": api.user_timeline(screen_name="Rager", count=50, lang="en", tweet_mode="extended"),
        "scottmelker": api.user_timeline(screen_name="scottmelker", count=50, lang="en", tweet_mode="extended"),
        "cameron": api.user_timeline(screen_name="cameron", count=50, lang="en", tweet_mode="extended"),
        # Food
        "Ottolenghi": api.user_timeline(screen_name="Ottolenghi", count=50, lang="en", tweet_mode="extended"),
        "Foodimentary": api.user_timeline(screen_name="Foodimentary", count=50, lang="en", tweet_mode="extended"),
        "SpoonUniversity": api.user_timeline(screen_name="SpoonUniversity", count=50, lang="en", tweet_mode="extended"),
        "epicurious": api.user_timeline(screen_name="epicurious", count=50, lang="en", tweet_mode="extended"),
        # Health and fitness
        "DailyFitnessTip": api.user_timeline(screen_name="DailyFitnessTip", count=50, lang="en", tweet_mode="extended"),
        "NutritionDiva": api.user_timeline(screen_name="NutritionDiva", count=50, lang="en", tweet_mode="extended"),
        "robbwolf": api.user_timeline(screen_name="robbwolf", count=50, lang="en", tweet_mode="extended"),
        "WebMD": api.user_timeline(screen_name="WebMD", count=50, lang="en", tweet_mode="extended"),
        # Tech
        "TechCrunch": api.user_timeline(screen_name="TechCrunch", count=50, lang="en", tweet_mode="extended"),
        "robbwolf": api.user_timeline(screen_name="robbwolf", count=50, lang="en", tweet_mode="extended"),
        "RyersonDMZ": api.user_timeline(screen_name="RyersonDMZ", count=50, lang="en", tweet_mode="extended"),
        "jeffweiner": api.user_timeline(screen_name="jeffweiner", count=50, lang="en", tweet_mode="extended"),
        # Music
        "future_of_music": api.user_timeline(screen_name="future_of_music", count=50, lang="en", tweet_mode="extended"),
        "MusicFactsFun": api.user_timeline(screen_name="MusicFactsFun", count=50, lang="en", tweet_mode="extended"),
        "gillespeterson": api.user_timeline(screen_name="gillespeterson", count=50, lang="en", tweet_mode="extended"),
        "ElliottWilson": api.user_timeline(screen_name="ElliottWilson", count=50, lang="en", tweet_mode="extended"),
        # Christian
        "SheReadsTruth": api.user_timeline(screen_name="SheReadsTruth", count=50, lang="en", tweet_mode="extended"),
        "biblegateway": api.user_timeline(screen_name="biblegateway", count=50, lang="en", tweet_mode="extended"),
        "timkellernyc": api.user_timeline(screen_name="timkellernyc", count=50, lang="en", tweet_mode="extended"),
        "JoelOsteen": api.user_timeline(screen_name="JoelOsteen", count=50, lang="en", tweet_mode="extended"),
        # Muslim
        "BonsaiSky": api.user_timeline(screen_name="BonsaiSky", count=50, lang="en", tweet_mode="extended"),
        "muftimenk": api.user_timeline(screen_name="muftimenk", count=50, lang="en", tweet_mode="extended"),
        "boonaamohammed": api.user_timeline(screen_name="boonaamohammed", count=50, lang="en", tweet_mode="extended"),
        "YasminMogahed": api.user_timeline(screen_name="YasminMogahed", count=50, lang="en", tweet_mode="extended"),
        # Buddhism
        "dzigarkongtrul": api.user_timeline(screen_name="dzigarkongtrul", count=50, lang="en", tweet_mode="extended"),
        "chademeng": api.user_timeline(screen_name="chademeng", count=50, lang="en", tweet_mode="extended"),
        "nobodhi": api.user_timeline(screen_name="nobodhi", count=50, lang="en", tweet_mode="extended"),
        "Buddhism_Now": api.user_timeline(screen_name="Buddhism_Now", count=50, lang="en", tweet_mode="extended")
    }

    with open('dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        head = ["Username", "Tweet"]
        writer.writerow(head)
        for k, v in tweets.items():
            username = k
            for tweet in v:
                text = tweet.full_text
                tw = [username, text]
                writer.writerow(tw)


def populate_database():
    with open("dataset.csv") as fp:
        reader = csv.reader(fp, delimiter=",")
        next(reader, None)
        data_read = [row for row in reader]

    print(data_read)

    for x in data_read:
        user = graph.nodes.match("User", username=x[0]).first()
        if not user:
            user = Node("User", username=x[0], password=bcrypt.encrypt("asd"))
            graph.create(user)
        user = User(x[0])
        tags = set({tag.strip("#") for tag in x[1].split() if tag.startswith("#")})
        # mentions = set({mention.strip("@") for mention in tweet.full_text.split() if mention.startswith("@")})
        user.add_post(tags, x[1])


def extract_entities():
    query = """
    CALL apoc.periodic.iterate(
      "MATCH (p:Filtered_Post)
        WHERE not(exists(p.processed))

       RETURN p",
      "CALL apoc.nlp.gcp.entities.stream([item in $_batch | item.p], {
         nodeProperty: 'text',
         key: $key
       })
       YIELD node, value
       SET node.processed = true
       WITH node, value
       UNWIND value.entities AS entity
       WITH entity, node
       WHERE not(entity.metadata.wikipedia_url is null)
       MERGE (ent:Entity {ent: entity.name, uri: entity.metadata.wikipedia_url})
       MERGE (node)-[:HAS_ENTITY]->(ent)",
      {batchMode: "BATCH_SINGLE", batchSize: 10, params: {key: "AIzaSyASU1vWgEd5KIi_xEwBC9Mz0BjhX9_vPNg"}})
    YIELD batches, total, timeTaken, committedOperations
    RETURN batches, total, timeTaken, committedOperations;
    """

    graph.run(query)


def filter_tweets(all):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')

    def clean_text(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Removes @mentions
        text = re.sub(r'#', '', text)  # Removes # symbol
        text = re.sub(r'RT[\s]+', '', text)  # Removing RT
        text = re.sub(r'https?:\/\/\S+', '', text)  # Removing links

        return text

    # https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    def demojify(text):
        regrex_pattern = re.compile(pattern="["
                                            u"\U0001F600-\U0001F64F"  # emoticons
                                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                            "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', text)

    posts = []
    ids = []
    string.punctuation = string.punctuation + '’'
    string.punctuation = string.punctuation + '…'
    string.punctuation = string.punctuation + '—'

    for i in all:
        # ’
        posts.append(demojify(clean_text(i['tweet'])))

    for i in all:
        ids.append(i['id'])

    total = len(posts)

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('wordnet')

    all_split_posts = []
    for i in range(0, total):
        split_post = posts[i].split(' ')
        split_post = [lemmatizer.lemmatize(x.translate(str.maketrans('', '', string.punctuation))) for x in split_post if
                      x.lower().islower()
                      ]
        all_split_posts.append(split_post)

    filtered_tweets = [' '.join(i) for i in all_split_posts]

    for i in range(0, total):
        a = graph.nodes.match("Post", id=ids[i]).first()
        b = Node("Filtered_Post", text=filtered_tweets[i])
        graph.create(b)
        graph.create(Relationship(a, "FILTERED", b))


# def link_categories():
#     query = """
#     MATCH (e:Entity), (c:Category)
#     WHERE e.ent = toLower(replace(c.catName, "_", " "))
#     CREATE (e)-[:IN_CATEGORY]->(c)
#     """

#     graph.run(query)


def link_entities():
    query = """
    MATCH (e:Entity), (p:Page)
    WHERE e.uri = p.uri
    CREATE (e)-[:ABOUT]->(p)
    """

    graph.run(query)


# create_dataset()
# populate_database()
# get_keywords(get_some_posts())
# link_keywords_to_wordnet()
# filter_tweets(get_all_posts())
# extract_entities()
# # link_categories()
# link_entities()

# link_keywords_to_wordnet()
# set_weights()
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
