from flask import Flask, request, session, redirect, url_for, render_template, flash
from .models import User
from .models import todays_recent_posts
from py2neo import Node, Graph
import time
from datetime import datetime
from datetime import timedelta
app = Flask(__name__)

graph = Graph('bolt://neo4j@localhost:7687', user="neo4j", password="123")


@app.route("/")
def index():
    if session.get("username") is not None:
        user = session["username"]
        start = float("%.20f" % time.time())
        posts = todays_recent_posts(5, user)
        end = float("%.20f" % time.time())
        print()
        print()
        print(end - start)
        print()
        print()
        return render_template("index.html", posts=posts)
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User(username)

        if not user.register(password):
            flash("A user with that username already exist")
        else:
            flash("You were successfuly registered")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User(username)

        if not user.verify_password(password):
            flash("Invalid login.")
        else:
            flash("Successful login.")
            session["username"] = user.username
            return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/add_post", methods=["POST"])
def add_post():
    text = request.form["text"]
    tags = set({tag.strip("#") for tag in text.split() if tag.startswith("#")})
    # mentions = set({mention.strip("@") for mention in text.split() if mention.startswith("@")})
    user = User(session["username"])
    # user.add_post(tags, mentions, text)
    user.add_post(tags, text)
    return redirect(url_for("index"))


# Adding tweets for experiment

# u_name = "ana"
# for u in range(0, 200):
#     u_name = u_name + "a"
#     user = Node("User", username=u_name, password="ana")
#     graph.create(user)
#     user = User(u_name)
#
#     for x in range(1, 5):
#         k = 1
#         while k < 5:
#             user.manual_post_add("Test post test post test post test post test post", x)
#             k += 1


@app.route("/like_post/<post_id>")
def like_post(post_id):
    username = session.get("username")
    if not username:
        flash("You must be logged in to like a post")
        return redirect(url_for("login"))

    user = User(username)
    user.like_post(post_id)
    flash("Post liked.")
    return redirect(request.referrer)


@app.route("/follow/<username>")
def follow(username):
    user1 = User(session["username"])
    user2 = User(username)
    user1.follow_user(user2)
    return redirect(request.referrer)


@app.route("/unfollow/<username>")
def unfollow(username):
    user1 = User(session["username"])
    user2 = User(username)
    user1.unfollow_user(user2)
    return redirect(request.referrer)


@app.route("/block/<username>")
def block(username):
    user1 = User(session["username"])
    user2 = User(username)
    user1.block_user(user2)
    return redirect(request.referrer)


@app.route("/unblock/<username>")
def unblock(username):
    user1 = User(session["username"])
    user2 = User(username)
    user1.unblock_user(user2)
    return redirect(request.referrer)


@app.route("/profile/<username>")
def profile(username):
    user1 = User(session.get("username"))
    user2 = User(username)
    follows = user1.is_following(username)
    blocks = user1.is_blocked(username)
    start = float("%.20f" % time.time())
    posts = user2.recent_posts(5)
    end = float("%.20f" % time.time())
    print()
    print()
    print(end-start)
    print()
    print()
    related_users = user1.get_accounts()
    following = len(related_users["following"])
    followers = len(related_users["followers"])

    similar = []
    common = {}

    if user1.username == user2.username:
        similar = user1.similar_users(3)
    else:
        common = user1.commonality_of_user(user2)

    return render_template("profile.html", username=username, posts=posts, similar=similar,
                           common=common, follows=follows, blocks=blocks, related_users=related_users,
                           following=following, followers=followers)


@app.route("/logout")
def logout():
    session.pop("username")
    flash("Logged out.")
    return redirect(url_for("index"))