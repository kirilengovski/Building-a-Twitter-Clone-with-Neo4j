from flask import Flask, request, session, redirect, url_for, render_template, flash
from .models import User
from .models import todays_recent_posts, tag_search

app = Flask(__name__)


@app.route("/")
def index():
    if session.get("username") is not None:
        user = session["username"]
        posts = todays_recent_posts(10, user)
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


@app.route("/search", methods=["GET", "POST"])
def search():
    username = session["username"]
    user = User(username)
    search = request.form["search"]
    users = user.get_users(search)
    if len(users) > 0:
        follows = user.is_following(users[0])
        blocks = user.is_blocked(username)

        return render_template("search.html", users=users, search=search, follows=follows, blocks=blocks)

    flash("No results.")
    return redirect(url_for("index"))


@app.route("/advanced_search", methods=["GET", "POST"])
def advanced_search():
    username = session["username"]
    user = User(username)
    search = request.form["advanced_search"]
    users = user.advanced_search(search)
    print(users)
    if len(users) > 0:
        follows = user.is_following(users[0])
        blocks = user.is_blocked(username)

        return render_template("search.html", users=users, search=search, follows=follows, blocks=blocks)

    flash("No results.")
    return redirect(url_for("index"))


@app.route("/add_post", methods=["POST"])
def add_post():
    text = request.form["text"]
    tags = set({tag.strip("#") for tag in text.split() if tag.startswith("#")})
    # mentions = set({mention.strip("@") for mention in text.split() if mention.startswith("@")})
    user = User(session["username"])
    user.add_post(tags, text)
    return redirect(url_for("index"))


@app.route("/like_post/<post_id>")
def like_post(post_id):
    username = session.get("username")
    if not username:
        flash("You must be logged in to like a post.")
        return redirect(url_for("login"))

    user = User(username)
    user.like_post(post_id)
    flash("Post liked.")
    return redirect(request.referrer)

@app.route("/retweet/<post_id>")
def retweet(post_id):
    username = session.get("username")
    if not username:
        flash("You must be logged in to retweet.")
        return redirect(url_for("login"))

    user = User(username)
    user.retweet(post_id)
    flash("Post retweeted.")
    return redirect(request.referrer)


@app.route("/undo_retweet/<post_id>")
def undo_retweet(post_id):
    username = session.get("username")
    user = User(username)
    user.undo_retweet(post_id)
    flash("Undo retweet successful.")
    return redirect(request.referrer)


@app.route("/follow/<username>")
def follow(username):
    user1 = User(session["username"])
    user2 = User(username)
    user1.follow_user(user2)
    flash("User followed.")
    return redirect(url_for("profile", username=username))


@app.route("/unfollow/<username>")
def unfollow(username):
    user1 = User(session["username"])
    user2 = User(username)
    user1.unfollow_user(user2)
    flash("User unfollowed.")
    return redirect(url_for("index"))


@app.route("/block/<username>")
def block(username):
    user1 = User(session["username"])
    user2 = User(username)
    user1.block_user(user2)
    flash("User blocked.")
    return redirect(url_for("index"))


@app.route("/unblock/<username>")
def unblock(username):
    user1 = User(session["username"])
    user2 = User(username)
    user1.unblock_user(user2)
    return redirect(request.referrer)


@app.route("/tag/<name>")
def tag(name):
    posts = tag_search(name, 6, session.get("username"))
    return render_template("tag.html", posts=posts, name=name)


@app.route("/profile/<username>")
def profile(username):
    user1 = User(session.get("username"))
    user2 = User(username)
    follows = user1.is_following(username)
    blocks = user1.is_blocked(username)
    posts = user2.recent_posts(6, session.get("username"))
    related_users = user1.get_accounts()
    recommended_users = user1.get_recommended_users()
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
                           following=following, followers=followers, recommended_users=recommended_users)


@app.route("/logout")
def logout():
    session.pop("username")
    flash("Logged out.")
    return redirect(url_for("index"))