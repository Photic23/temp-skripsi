"""
annotate_app.py
Flask web app for multi-annotator forum summarization.

Usage:
    python annotate_app.py          # runs on http://localhost:5001
"""
import json
import os
import threading
import tempfile
from pathlib import Path
from functools import wraps

from flask import Flask, render_template, request, session, redirect, url_for, flash
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "annotation-secret-key-change-me")

DATA_FILE = Path("web_annotation.json")
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data() -> list[dict]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def save_data(data: list[dict]):
    """Atomic write to prevent corruption if two annotators save simultaneously."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=".", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, DATA_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise


def get_annotators() -> dict[str, str]:
    """Returns {name: password} from .env."""
    result = {}
    for i in range(1, 10):
        name = os.getenv(f"ANNOTATOR_{i}_NAME")
        pwd = os.getenv(f"ANNOTATOR_{i}_PASS")
        if name and pwd:
            result[name] = pwd
    return result


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "annotator" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def find_entry(data: list[dict], forum_id: str) -> dict | None:
    for entry in data:
        if entry["id"] == forum_id:
            return entry
    return None


def completion_stats(data: list[dict], annotator: str) -> tuple[int, int]:
    done = sum(1 for e in data if e["annotations"].get(annotator, "").strip())
    return done, len(data)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def login():
    if "annotator" in session:
        return redirect(url_for("forum_list"))

    annotators = get_annotators()
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        pwd = request.form.get("password", "")
        if name in annotators and annotators[name] == pwd:
            session["annotator"] = name
            return redirect(url_for("forum_list"))
        error = "Nama atau password salah."

    return render_template("login.html", annotator_names=list(annotators.keys()), error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/forums")
@login_required
def forum_list():
    annotator = session["annotator"]
    data = load_data()
    done, total = completion_stats(data, annotator)

    entries = [
        {
            "id": e["id"],
            "title": e["title"],
            "course": e["course"],
            "total_posts": e["total_posts"],
            "done": bool(e["annotations"].get(annotator, "").strip()),
        }
        for e in data
    ]
    return render_template("forum_list.html", entries=entries, annotator=annotator,
                           done=done, total=total)


@app.route("/forums/<path:forum_id>", methods=["GET", "POST"])
@login_required
def forum_detail(forum_id):
    annotator = session["annotator"]

    with _lock:
        data = load_data()
        entry = find_entry(data, forum_id)
        if entry is None:
            flash("Forum tidak ditemukan.")
            return redirect(url_for("forum_list"))

        if request.method == "POST":
            summary = request.form.get("summary", "").strip()
            entry["annotations"][annotator] = summary
            save_data(data)
            flash("Anotasi tersimpan!", "success")

            # Redirect to next unannotated forum
            ids = [e["id"] for e in data]
            current_idx = ids.index(forum_id)
            for i in range(current_idx + 1, len(data)):
                if not data[i]["annotations"].get(annotator, "").strip():
                    return redirect(url_for("forum_detail", forum_id=data[i]["id"]))
            # All done
            return redirect(url_for("forum_list"))

    existing = entry["annotations"].get(annotator, "")
    done, total = completion_stats(data, annotator)

    # Prev / next navigation
    ids = [e["id"] for e in data]
    idx = ids.index(forum_id)
    prev_id = ids[idx - 1] if idx > 0 else None
    next_id = ids[idx + 1] if idx < len(ids) - 1 else None

    return render_template(
        "forum.html",
        entry=entry,
        annotator=annotator,
        existing_summary=existing,
        done=done,
        total=total,
        prev_id=prev_id,
        next_id=next_id,
        current_num=idx + 1,
    )


@app.route("/admin/progress")
@login_required
def progress():
    data = load_data()
    annotators = list(get_annotators().keys())
    rows = []
    for entry in data:
        row = {
            "id": entry["id"],
            "title": entry["title"],
            "statuses": {a: bool(entry["annotations"].get(a, "").strip()) for a in annotators},
        }
        rows.append(row)
    totals = {a: sum(1 for r in rows if r["statuses"][a]) for a in annotators}
    return render_template("progress.html", rows=rows, annotators=annotators,
                           totals=totals, total=len(data))


if __name__ == "__main__":
    if not DATA_FILE.exists():
        print("web_annotation.json not found. Run migrate_annotation.py first.")
        raise SystemExit(1)
    if not get_annotators():
        print("No annotators configured. Add ANNOTATOR_1_NAME / ANNOTATOR_1_PASS to .env")
        raise SystemExit(1)
    app.run(debug=True, port=5001)
