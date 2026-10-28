import os
import re
import hmac
import secrets
import sqlite3
from pathlib import Path
from functools import wraps
from flask import (
    render_template, request, redirect, url_for, flash,
    session, g
)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
USER_DB = DATA_DIR / "users.db"


def _connect():
    db = sqlite3.connect(USER_DB)
    db.row_factory = sqlite3.Row
    return db


def init_auth(app):
    # Use a stable random key unless the user explicitly provides one.
    secret_env = os.getenv("FLASK_SECRET_KEY")
    secret_file = DATA_DIR / ".flask_secret_key"

    if secret_env:
        app.secret_key = secret_env
    else:
        if not secret_file.exists():
            secret_file.write_text(secrets.token_hex(32))
            try:
                secret_file.chmod(0o600)
            except OSError:
                pass
        app.secret_key = secret_file.read_text().strip()

    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        MAX_CONTENT_LENGTH=50 * 1024 * 1024,
    )

    with _connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def csrf_token():
        if "_csrf_token" not in session:
            session["_csrf_token"] = secrets.token_urlsafe(32)
        return session["_csrf_token"]

    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def protect_workspace():
        # Protect existing application tools without changing their routes.
        public_endpoints = {
            "index", "login", "register", "health", "static"
        }
        if request.endpoint is None or request.endpoint in public_endpoints:
            return None

        if not session.get("user_id"):
            flash("Please sign in to access your workspace.", "info")
            return redirect(url_for("login", next=request.full_path))

        return None

    def valid_csrf():
        supplied = request.form.get("_csrf_token", "")
        expected = session.get("_csrf_token", "")
        return bool(supplied and expected and hmac.compare_digest(supplied, expected))

    def safe_next(value):
        # Only allow local relative URLs to avoid open redirects.
        if value and value.startswith("/") and not value.startswith("//"):
            return value
        return url_for("workspace")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if session.get("user_id"):
            return redirect(url_for("workspace"))

        if request.method == "POST":
            if not valid_csrf():
                flash("Your form expired. Please try again.", "error")
                return redirect(url_for("login"))

            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            with _connect() as db:
                user = db.execute(
                    "SELECT id, username, password_hash FROM users WHERE username = ?",
                    (username,)
                ).fetchone()

            if user and check_password_hash(user["password_hash"], password):
                session.clear()
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                flash("Welcome back.", "success")
                return redirect(safe_next(request.form.get("next", "")))

            flash("Invalid username or password.", "error")

        return render_template("login.html", next_url=safe_next(request.args.get("next", "")))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if session.get("user_id"):
            return redirect(url_for("workspace"))

        if request.method == "POST":
            if not valid_csrf():
                flash("Your form expired. Please try again.", "error")
                return redirect(url_for("register"))

            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm_password", "")

            if not re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", username):
                flash("Username must be 3–30 characters: letters, numbers, dots, hyphens or underscores.", "error")
            elif len(password) < 8:
                flash("Use a password with at least 8 characters.", "error")
            elif password != confirm:
                flash("The passwords do not match.", "error")
            else:
                try:
                    with _connect() as db:
                        cur = db.execute(
                            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                            (username, generate_password_hash(password))
                        )
                        user_id = cur.lastrowid

                    session.clear()
                    session["user_id"] = user_id
                    session["username"] = username
                    flash("Your account has been created.", "success")
                    return redirect(url_for("workspace"))
                except sqlite3.IntegrityError:
                    flash("That username is already registered.", "error")

        return render_template("register.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        if not valid_csrf():
            flash("Your form expired. Please try again.", "error")
            return redirect(url_for("workspace"))

        session.clear()
        flash("You have signed out.", "success")
        return redirect(url_for("index"))

    @app.route("/workspace")
    def workspace():
        return render_template("workspace.html")
