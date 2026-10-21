

import streamlit as st

import pandas as pd

import joblib
import base64
import sqlite3
import json
import hashlib
import hmac
import secrets

from pathlib import Path

from datetime import datetime

# -----------------------------

# 1. Project files

# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"

MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"

DATA_PATH = BASE_DIR / "data" / "student_data.csv"


# -----------------------------
# Persistent report database (SQLite)
# -----------------------------
DATABASE_PATH = BASE_DIR / "data" / "student_reports.db"
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


def hash_password(password, salt=None):
    """Hash a password with PBKDF2; never store plain-text passwords."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    ).hex()
    return f"{salt}${digest}"


def verify_password(password, stored_hash):
    try:
        salt, expected_digest = stored_hash.split("$", 1)
        actual_digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
        ).hex()
        return hmac.compare_digest(actual_digest, expected_digest)
    except (ValueError, AttributeError):
        return False


def get_student_account(student_id):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT student_id, name, password_hash, email, department "
            "FROM student_accounts WHERE student_id = ?",
            (str(student_id),),
        ).fetchone()
    return dict(row) if row else None


def get_all_student_accounts():
    """Return account profile fields only; never return password hashes."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT student_id, name, email, department, created_at
            FROM student_accounts
            ORDER BY name COLLATE NOCASE
            """
        ).fetchall()
    return [dict(row) for row in rows]


def reset_student_password(student_id, new_password):
    """Set a new password for a student; only the hash is stored."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            "UPDATE student_accounts SET password_hash = ? WHERE student_id = ?",
            (hash_password(new_password), str(student_id)),
        )
        connection.commit()
        return cursor.rowcount > 0


def create_student_account(student_id, name, password, email="", department=""):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "INSERT INTO student_accounts "
            "(student_id, name, password_hash, email, department, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(student_id), name.strip(), hash_password(password),
                email.strip(), department,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        connection.commit()


def init_student_accounts():
    """Create persistent accounts and seed demo users only if missing."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS student_accounts (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT NOT NULL DEFAULT '',
                department TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )"""
        )
        connection.commit()

    demo_accounts = [
        ("1234567", "Ali Ahmed", "demo123"),
        ("2345678", "Sara Khan", "demo123"),
        ("3456789", "Hamza Malik", "demo123"),
    ]
    # Keep reserved demo accounts aligned with the credentials shown on the login page.
    # These IDs are for testing only and are reset to demo123 whenever the app starts.
    for student_id, name, password in demo_accounts:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                """
                INSERT INTO student_accounts
                    (student_id, name, password_hash, email, department, created_at)
                VALUES (?, ?, ?, '', '', ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    name = excluded.name,
                    password_hash = excluded.password_hash
                """,
                (
                    student_id,
                    name,
                    hash_password(password),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            connection.commit()


def init_report_database():
    """Create the reports table if it does not exist yet."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS student_reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                report_type TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS student_login_history (
                login_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                login_time TEXT NOT NULL,
                login_status TEXT NOT NULL DEFAULT 'Successful'
            )
            """
        )
        # Permanent audit trail for removed students. The account can be deleted
        # while an administrator-only record preserves the reason and key history.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS removed_students (
                removal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                email TEXT NOT NULL DEFAULT '',
                department TEXT NOT NULL DEFAULT '',
                account_created_at TEXT NOT NULL DEFAULT '',
                removed_at TEXT NOT NULL,
                removed_by TEXT NOT NULL,
                remarks TEXT NOT NULL,
                reports_count INTEGER NOT NULL DEFAULT 0,
                login_count INTEGER NOT NULL DEFAULT 0,
                archived_reports_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        connection.commit()


def save_report_to_database(report, report_type="prediction"):
    """Save one report persistently and associate it with a Student ID."""
    student_id = str(report.get("Student ID", "")).strip()
    student_name = str(report.get("Student Name", "")).strip()
    if not student_id or not student_name:
        raise ValueError("Report must contain Student ID and Student Name.")

    created_at = str(
        report.get("Report Generated")
        or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO student_reports
                (student_id, student_name, report_type, report_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                student_id,
                student_name,
                report_type,
                json.dumps(report, ensure_ascii=False),
                created_at,
            ),
        )
        connection.commit()


def get_reports_from_database(student_id=None):
    """Return persistent reports; optionally limit results to one Student ID."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        if student_id is None:
            rows = connection.execute(
                "SELECT report_json FROM student_reports ORDER BY report_id DESC"
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT report_json FROM student_reports
                WHERE student_id = ?
                ORDER BY report_id DESC
                """,
                (str(student_id),),
            ).fetchall()
    return [json.loads(row[0]) for row in rows]


def record_student_login(student_id, student_name):
    """Record a successful student login without storing credentials."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "INSERT INTO student_login_history (student_id, student_name, login_time) VALUES (?, ?, ?)",
            (str(student_id), str(student_name), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        connection.commit()


def get_student_login_history():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT student_id, student_name, login_time, login_status FROM student_login_history ORDER BY login_id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_removed_students():
    """Return administrator-only removal audit records, newest first."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT removal_id, student_id, student_name, email, department,
                      account_created_at, removed_at, removed_by, remarks,
                      reports_count, login_count, archived_reports_json
               FROM removed_students ORDER BY removal_id DESC"""
        ).fetchall()
    records = []
    for row in rows:
        item = dict(row)
        try:
            item["archived_reports"] = json.loads(item.pop("archived_reports_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            item["archived_reports"] = []
            item.pop("archived_reports_json", None)
        records.append(item)
    return records


def delete_student_account(student_id, remarks, removed_by="Administrator"):
    """Archive student details and reason, then delete active account and live records."""
    sid = str(student_id).strip()
    reason = str(remarks).strip()
    if not reason:
        raise ValueError("Removal remarks are required.")

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        account = connection.execute(
            "SELECT student_id, name, email, department, created_at "
            "FROM student_accounts WHERE TRIM(student_id) = ?", (sid,)
        ).fetchone()
        if account is None:
            return False

        report_rows = connection.execute(
            "SELECT report_json FROM student_reports WHERE TRIM(student_id) = ? ORDER BY report_id",
            (sid,),
        ).fetchall()
        login_rows = connection.execute(
            "SELECT student_name, login_time, login_status FROM student_login_history "
            "WHERE TRIM(student_id) = ? ORDER BY login_id", (sid,)
        ).fetchall()
        archived_reports = []
        for row in report_rows:
            try:
                archived_reports.append(json.loads(row[0]))
            except (TypeError, json.JSONDecodeError):
                archived_reports.append({"raw_report": row[0]})

        removed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection.execute(
            """INSERT INTO removed_students
               (student_id, student_name, email, department, account_created_at,
                removed_at, removed_by, remarks, reports_count, login_count,
                archived_reports_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                account["student_id"], account["name"], account["email"],
                account["department"], account["created_at"], removed_at,
                str(removed_by).strip() or "Administrator", reason,
                len(report_rows), len(login_rows),
                json.dumps(archived_reports, ensure_ascii=False),
            ),
        )
        connection.execute("DELETE FROM student_reports WHERE TRIM(student_id) = ?", (sid,))
        connection.execute("DELETE FROM student_login_history WHERE TRIM(student_id) = ?", (sid,))
        connection.execute("DELETE FROM student_accounts WHERE TRIM(student_id) = ?", (sid,))
        connection.commit()

    # Remove matching data from the current Streamlit session cache too.
    cached_reports = st.session_state.get("student_reports", [])
    if isinstance(cached_reports, list):
        st.session_state.student_reports = [
            report for report in cached_reports
            if str(report.get("Student ID", report.get("student_id", ""))).strip() != sid
        ]
    current_report = st.session_state.get("student_report")
    if isinstance(current_report, dict) and str(current_report.get("Student ID", "")).strip() == sid:
        st.session_state.pop("student_report", None)
    return True


init_report_database()
init_student_accounts()

# -----------------------------

# 2. Website configuration

# -----------------------------

st.set_page_config(

    page_title="Student Performance System",

    page_icon="🎓",

    layout="wide"

)

# ==========================================

# STUDENT LOGIN AND REGISTRATION

# ==========================================

if "student_logged_in" not in st.session_state:

    st.session_state.student_logged_in = False

if "current_student" not in st.session_state:

    st.session_state.current_student = None

# Show login page when student is not logged in

if not st.session_state.student_logged_in:

    st.markdown("""

    <style>

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {

        background: linear-gradient(135deg, #eaf2ff 0%, #f7f9ff 55%, #dce8ff 100%) !important;

    }

    [data-testid="stHeader"] { background: transparent !important; }

    .block-container { max-width: 1180px; padding-top: 1.5rem; padding-bottom: 2rem; }

    h1, h2, h3, h4 { color: #173b70 !important; font-weight: 750 !important; }

    [data-testid="stMarkdownContainer"] p,

    [data-testid="stCaptionContainer"],

    [data-testid="stWidgetLabel"], label,

    [data-testid="stText"] { color: #243247 !important; }

    [data-testid="stTextInput"] input,

    [data-testid="stNumberInput"] input {

        color: #172033 !important; background: #ffffff !important;

        border: 1px solid #b8cbed !important; border-radius: 10px !important;

        min-height: 42px;

    }

    [data-testid="stFormSubmitButton"] button {

        background: #2458c6 !important; color: #ffffff !important;

        border: none !important; border-radius: 10px !important;

        min-height: 44px; font-weight: 700 !important;

    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {

        background: #dce8fb !important; border-radius: 12px; padding: 6px; gap: 8px;

    }

    [data-testid="stTabs"] button[role="tab"] { color: #173b70 !important; }

    [data-testid="stAlert"] { border-radius: 12px; }

    </style>

    """, unsafe_allow_html=True)

    st.title("🎓 Student Portal")

    st.subheader("Your Journey. Your Success.")

    st.write("Sign in to view your student dashboard, reports and performance analysis.")

    login_tab, register_tab, admin_tab = st.tabs([
    "🔐 Student Login",
    "📝 Create Account",
    "🔑 Admin Login"
])

    # LOGIN PAGE: image on the left, login details on the right

    with login_tab:

        image_col, form_col = st.columns([1, 1], gap="large", vertical_alignment="center")

        with image_col:

            login_image = IMAGE_DIR / "login2.jpg"

            if login_image.exists():

                st.image(str(login_image), width="stretch")

            else:

                st.info("Add login2.jpg to the project's images folder to show the login picture.")

            st.markdown("### Learn. Grow. Succeed.")

            st.write("Your academic journey starts here.")

            st.caption("📊 Performance Analysis  |  🎯 Score Prediction")

        with form_col:

            st.subheader("Welcome Back!")

            st.write("Enter your Student ID and password to continue.")

            with st.form("student_login_form"):

                login_id = st.text_input("Student ID", key="login_student_id")

                login_password = st.text_input("Password", type="password", key="login_student_password")

                login_button = st.form_submit_button("Login", width="stretch")

            if login_button:

                sid = login_id.strip()

                if not sid.isdigit() or len(sid) < 7:

                    st.error("Student ID must contain at least 7 numeric digits.")

                else:

                    user = get_student_account(sid)

                    if user and verify_password(login_password, user["password_hash"]):

                        st.session_state.student_logged_in = True
                        st.session_state.user_role = "student"
                        st.session_state.current_student = {
                            "id": sid,
                            "name": user["name"],
                            "email": user.get("email", ""),
                            "department": user.get("department", ""),
                        }
                        record_student_login(sid, user["name"])

                        st.rerun()

                    else:

                        st.error("Student Not Found: Invalid ID or password.")

            st.info("Demo login: ID 1234567  |  Password: demo123 (testing only)")

    # REGISTRATION PAGE: a separate image plus student profile details

    
    # REGISTRATION PAGE: full background image
    with register_tab:
        register_image = IMAGE_DIR / "login1.jpg"

        if register_image.exists():
            image_bytes = register_image.read_bytes()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            st.markdown(
                f"""
                <style>
                .st-key-register_page {{
                    background-image:
                        linear-gradient(
                            rgba(10, 25, 55, 0.30),
                            rgba(10, 25, 55, 0.30)
                        ),
                        url("data:image/jpeg;base64,{image_base64}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    min-height: 85vh;
                    padding: 28px;
                    border-radius: 18px;
                }}

                .st-key-register_form_card {{
                    background: rgba(255, 255, 255, 0.94);
                    padding: 24px;
                    border-radius: 16px;
                    border: 1px solid rgba(255,255,255,0.7);
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error(
                "Image not found. Make sure images/login1.jpg exists."
            )

        with st.container(key="register_page"):
            st.title("🎓 Start Your Learning Journey")
            st.write(
                "Create your student profile and access your academic dashboard."
            )

            left_col, right_col = st.columns(
                [0.8, 1.2],
                gap="large"
            )

            with left_col:
                st.markdown("### Learn. Grow. Succeed.")
                st.write(
                    "Your academic journey starts here."
                )
                st.write(
                    "📚 Learning Progress  |  🎯 Student Success"
                )

            with right_col:
                with st.container(key="register_form_card"):
                    st.subheader("Create Your Student Account")
                    st.caption(
                        "Enter your details below to register."
                    )

                    with st.form("student_registration_form"):
                        new_name = st.text_input("Full Name")
                        new_id = st.text_input(
                            "Create Student ID (minimum 7 digits)"
                        )
                        new_email = st.text_input(
                            "Email Address (optional)"
                        )

                        new_department = st.selectbox(
                            "Class / Department",
                            [
                                "Select an option",
                                "Computer Science",
                                "Information Technology",
                                "Software Engineering",
                                "Other"
                            ]
                        )

                        new_password = st.text_input(
                            "Create Password",
                            type="password"
                        )
                        confirm_password = st.text_input(
                            "Confirm Password",
                            type="password"
                        )

                        register_button = st.form_submit_button(
                            "Create Account",
                            width="stretch"
                        )

                    if register_button:
                        sid = new_id.strip()

                        if not new_name.strip():
                            st.error("Please enter your full name.")

                        elif not sid.isdigit() or len(sid) < 7:
                            st.error(
                                "Student ID must contain at least 7 numeric digits."
                            )

                        elif get_student_account(sid) is not None:
                            st.error(
                                "This Student ID already exists. Choose another."
                            )

                        elif get_reports_from_database(student_id=sid):
                            st.error(
                                "Reports already exist for this Student ID, but no account is registered. "
                                "Please ask an administrator to verify the account before proceeding."
                            )

                        elif new_email.strip() and (
                            "@" not in new_email
                            or "." not in new_email.split("@")[-1]
                        ):
                            st.error(
                                "Enter a valid email address or leave it empty."
                            )

                        elif new_department == "Select an option":
                            st.error(
                                "Please select your class / department."
                            )

                        elif len(new_password) < 6:
                            st.error(
                                "Password must be at least 6 characters."
                            )

                        elif new_password != confirm_password:
                            st.error("Passwords do not match.")

                        else:
                            try:
                                create_student_account(
                                    sid,
                                    new_name.strip(),
                                    new_password,
                                    new_email.strip(),
                                    new_department,
                                )
                                st.session_state.student_logged_in = True
                                st.session_state.user_role = "student"
                                st.session_state.current_student = {
                                    "id": sid,
                                    "name": new_name.strip(),
                                    "email": new_email.strip(),
                                    "department": new_department,
                                }
                                st.success("Account created successfully.")
                                st.rerun()
                            except sqlite3.IntegrityError:
                                st.error("This Student ID already exists. Choose another.")
                            except Exception as error:
                                st.error(f"Could not save the account: {error}")


    # ADMIN LOGIN
    with admin_tab:
        st.subheader("Administrator Login")
        st.write("Authorized administrators only.")

        with st.form("admin_login_form"):
            admin_username = st.text_input(
                "Admin Username",
                key="admin_login_username"
            )
            admin_password = st.text_input(
                "Admin Password",
                type="password",
                key="admin_login_password"
            )
            admin_button = st.form_submit_button(
                "Admin Login",
                width="stretch"
            )

        if admin_button:
            try:
                saved_username = st.secrets["admin"]["username"]
                saved_password = st.secrets["admin"]["password"]

                if (
                    admin_username.strip() == saved_username
                    and admin_password == saved_password
                ):
                    st.session_state.student_logged_in = True
                    st.session_state.user_role = "admin"
                    st.session_state.current_student = {
                        "id": "ADMIN",
                        "name": "Administrator"
                    }
                    st.rerun()
                else:
                    st.error("Invalid administrator username or password.")

            except (KeyError, FileNotFoundError):
                st.error(
                    "Admin credentials not found. "
                    "Check .streamlit/secrets.toml."
                )



    st.divider()

    st.caption("Demo accounts are for testing only. Student accounts are stored in the local SQLite database.")

    st.stop()

# Sidebar student information and logout

with st.sidebar:

    current_student = st.session_state.get("current_student")

    if current_student is not None:
        st.success(
            "Logged in: " + str(current_student.get("name", "Student"))
        )

        st.caption(
            "Student ID: " + str(current_student.get("id", "N/A"))
        )

        if st.button("🚪 Logout", width="stretch"):
         st.session_state.student_logged_in = False
         st.session_state.current_student = None
         st.session_state.user_role = None
         st.session_state.pop("student_report", None)
         st.session_state.pop("main_navigation", None)
         st.rerun()


    else:
        st.warning("Please login first.")

    # Simple role-based navigation keeps the main screen uncluttered.
    if current_student is not None:
        st.divider()
        st.subheader("🧭 Navigation")
        if st.session_state.get("user_role") == "admin":
            page_options = [
                "Dashboard",
                "Student Learning Gallery",
                "Student Accounts",
                "Removed Students",
                "Login History",
                "Student Reports",
            ]
        else:
            page_options = ["Dashboard", "My Reports", "Student Learning Gallery"]
        previous_page = st.session_state.get("main_navigation", "Dashboard")
        if previous_page not in page_options:
            previous_page = "Dashboard"
        selected_page = st.radio(
            "Go to page",
            page_options,
            index=page_options.index(previous_page),
            key="main_navigation",
            label_visibility="collapsed",
        )
    else:
        selected_page = "Dashboard"

# Main dashboard styling: keep the existing blue theme while improving spacing and visual hierarchy.
st.markdown("""
<style>
/* Layout and component styling; respect Streamlit's active theme variables */
.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1, h2, h3 {
    letter-spacing: -0.025em;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #4A102A 0%, #7F1D45 58%, #BE185D 100%);
    border: 1px solid #FB7185;
    padding: 17px 18px;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(74, 16, 42, 0.16);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #FFFFFF !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 650 !important;
}
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--border-color, #dbe6fb);
}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 10px 10px 0 0;
    font-weight: 650;
}
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
    border-radius: 10px;
    font-weight: 650;
    transition: all 0.15s ease-in-out;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border-color, #dbe6fb);
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stAlert"] {
    border-radius: 12px;
}
hr {
    border-color: var(--border-color, #dbe6fb);
}

/* Midnight Purple hero banner */
.dashboard-hero {
    position: relative;
    overflow: hidden;
    padding: 27px 30px;
    margin: 0 0 20px 0;
    border-radius: 22px;
    background: linear-gradient(115deg, #24104F 0%, #6D28D9 58%, #A78BFA 100%);
    box-shadow: 0 12px 28px rgba(109, 40, 217, 0.22);
    color: #FFFFFF;
}
.dashboard-hero:after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -60px;
    top: -90px;
    border-radius: 50%;
    background: rgba(255,255,255,0.10);
}
.dashboard-hero h2 {
    color: #FFFFFF !important;
    margin: 0 0 7px 0;
    font-size: clamp(1.45rem, 2.4vw, 2.15rem);
    font-weight: 800;
}
.dashboard-hero p {
    color: #F3E8FF !important;
    margin: 0;
    max-width: 760px;
    line-height: 1.55;
}
.hero-pill {
    display: inline-block;
    margin-bottom: 12px;
    padding: 5px 11px;
    border: 1px solid rgba(255,255,255,0.35);
    background: rgba(255,255,255,0.12);
    border-radius: 999px;
    color: #FFFFFF;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Burgundy & Rose cards, with high-contrast text in either Streamlit theme */
.feature-card {
    padding: 17px 18px;
    height: 100%;
    background: linear-gradient(145deg, #4A102A 0%, #8B1748 58%, #BE185D 100%);
    border: 1px solid #FB7185;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(74, 16, 42, 0.16);
    color: #FFFFFF;
}
.feature-card .feature-icon {
    font-size: 1.55rem;
    margin-bottom: 7px;
}
.feature-card h4 {
    margin: 0 0 5px 0;
    color: #FFFFFF !important;
    font-size: 1.02rem;
}
.feature-card p {
    margin: 0;
    color: #FFE4E6 !important;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Navy & Cyan sidebar. Inputs retain Streamlit's native theme styling. */
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #0B1F3A 0%, #075985 70%, #0E7490 100%);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #F0FDFF;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(34, 211, 238, 0.38);
}
[data-testid="stSidebar"] [data-testid="stButton"] button {
    border-color: #22D3EE;
}
@media (max-width: 700px) {
    .dashboard-hero { padding: 22px 20px; border-radius: 17px; }
    [data-testid="stMetric"] { padding: 12px; }
}
</style>
""", unsafe_allow_html=True)

if selected_page == "Dashboard":
    st.markdown("""
    <div class="dashboard-hero">
      <span class="hero-pill">Student Success Dashboard</span>
      <h2>🎓 Student Performance Prediction System</h2>
      <p>Turn academic data into meaningful insights. Explore score predictions, review reports, and track performance with confidence.</p>
    </div>
    """, unsafe_allow_html=True)

    feature_cols = st.columns(3, gap="medium")
    feature_content = [
        ("📈", "Performance Insights", "Explore learning patterns and understand the factors linked to exam scores."),
        ("🎯", "Score Prediction", "Estimate academic outcomes using the available student performance model."),
        ("📋", "Personal Reports", "Review saved reports and keep student information organized."),
    ]
    for feature_col, (feature_icon, feature_title, feature_description) in zip(feature_cols, feature_content):
        with feature_col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{feature_icon}</div>'
                f'<h4>{feature_title}</h4><p>{feature_description}</p></div>',
                unsafe_allow_html=True
            )

st.write("")

# Reports remain available during this browser session. Persistent storage requires a database.
if "student_reports" not in st.session_state:
    st.session_state.student_reports = []


import pandas as pd
import streamlit as st

if st.session_state.get("user_role") == "admin" and selected_page == "Student Reports":
    if st.session_state.get("user_role") == "admin":
        # Load student dataset

        try:
            df = pd.read_csv(DATA_PATH)

            # Check required columns
            required_columns = [
                "study_hours",
                "attendance",
                "previous_marks",
                "assignments_completed",
                "sleep_hours",
                "exam_score"
            ]

            if all(col in df.columns for col in required_columns):

                # Create report-only student IDs
                df = df.copy()
                df["student_id"] = [
                    f"STU-{i:03d}" for i in range(1, len(df) + 1)
                ]

                # Pass mark: 50 out of 100
                df["result"] = df["exam_score"].apply(
                    lambda score: "Pass" if score >= 50 else "Fail"
                )

                st.divider()
                st.header("🎓 Student Reports")

                # Search student report
                selected_id = st.selectbox(
                    "Select Student",
                    df["student_id"].tolist()
                )

                student = df[df["student_id"] == selected_id].iloc[0]

                col1, col2, col3 = st.columns(3)

                col1.metric("Exam Score", f"{student['exam_score']}/100")
                col2.metric("Result", student["result"])
                col3.metric("Attendance", f"{student['attendance']}%")

                st.subheader("📋 Individual Student Details")

                report_columns = [
                    "student_id",
                    "study_hours",
                    "attendance",
                    "previous_marks",
                    "assignments_completed",
                    "sleep_hours",
                    "exam_score",
                    "result"
                ]

                st.dataframe(
                    student[report_columns].to_frame("Student Details"),
                    width="stretch"
                )

                # Download individual report
                report_csv = student[report_columns].to_frame().T.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Student Report",
                    data=report_csv,
                    file_name=f"{selected_id}_report.csv",
                    mime="text/csv"
                )

                st.divider()
                st.header("✅ Pass/Fail Analysis")

                total_students = len(df)
                passed = int((df["result"] == "Pass").sum())
                failed = int((df["result"] == "Fail").sum())

                pass_rate = (passed / total_students * 100) if total_students else 0
                fail_rate = (failed / total_students * 100) if total_students else 0

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Records", total_students)
                col2.metric("Passed", passed)
                col3.metric("Failed", failed)

                st.write(f"**Pass Rate:** {pass_rate:.1f}%")
                st.progress(pass_rate / 100)

                st.write(f"**Fail Rate:** {fail_rate:.1f}%")
                st.progress(fail_rate / 100)

                st.subheader("📊 Pass vs Fail")
                result_counts = (
                    df["result"]
                    .value_counts()
                    .reindex(["Pass", "Fail"], fill_value=0)
                )

                st.bar_chart(result_counts)

                st.subheader("📑 Complete Student Results — Practice Dataset")
                st.dataframe(
                    df[report_columns],
                    width="stretch",
                    hide_index=True
                )

                # Also show reports generated for newly registered student accounts.
                saved_report_rows = get_reports_from_database()
                if saved_report_rows:
                    saved_reports_df = pd.DataFrame(saved_report_rows)
                    st.subheader("🧑‍🎓 Results for Registered Students")
                    st.caption("These reports come from student accounts and reports created through the portal.")
                    st.dataframe(saved_reports_df, width="stretch", hide_index=True)
                    combined_download_df = pd.concat(
                        [df[report_columns].rename(columns={
                            "student_id": "Student ID", "study_hours": "Study Hours",
                            "attendance": "Attendance (%)", "previous_marks": "Previous Marks",
                            "assignments_completed": "Assignments Completed", "sleep_hours": "Sleep Hours",
                            "exam_score": "Exam Score", "result": "Result"
                        }), saved_reports_df], ignore_index=True, sort=False
                    )
                else:
                    combined_download_df = df[report_columns].rename(columns={
                        "student_id": "Student ID", "study_hours": "Study Hours",
                        "attendance": "Attendance (%)", "previous_marks": "Previous Marks",
                        "assignments_completed": "Assignments Completed", "sleep_hours": "Sleep Hours",
                        "exam_score": "Exam Score", "result": "Result"
                    })
                    st.info("No portal reports have been saved yet. New student reports will appear here after they are generated.")

                st.download_button(
                    "⬇️ Download Complete Student Results (CSV)",
                    data=combined_download_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="complete_student_results.csv",
                    mime="text/csv"
                )

                st.divider()
                st.header("📈 Performance Analysis")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Average Exam Score",
                    f"{df['exam_score'].mean():.1f}/100"
                )

                col2.metric(
                    "Highest Score",
                    f"{df['exam_score'].max()}/100"
                )

                col3.metric(
                    "Lowest Score",
                    f"{df['exam_score'].min()}/100"
                )

                st.subheader("📚 Study Hours vs Exam Score")
                st.scatter_chart(
                    df,
                    x="study_hours",
                    y="exam_score"
                )

                st.subheader("📅 Attendance vs Exam Score")
                st.scatter_chart(
                    df,
                    x="attendance",
                    y="exam_score"
                )

                st.subheader("📝 Previous Marks vs Exam Score")
                st.scatter_chart(
                    df,
                    x="previous_marks",
                    y="exam_score"
                )

            else:
                st.error(
                    "Required columns are missing from the dataset. "
                    "Please check the column names in student_data.csv."
                )

        except FileNotFoundError:
            st.error(
                "Dataset not found. Make sure the file "
                "exists at 'data/student_data.csv'."
            )

        # -----------------------------

# 3. Load model and dataset

# -----------------------------

@st.cache_resource

def load_model():

    return joblib.load(MODEL_PATH)

@st.cache_data

def load_data():

    return pd.read_csv(DATA_PATH)

try:

    saved_model = load_model()

    model = saved_model["model"]

    features = saved_model["features"]

    df = load_data()

except Exception as error:

    st.error(f"Could not load project files: {error}")

    st.stop()

    # Set default user role
if "user_role" not in st.session_state:
    st.session_state.user_role = None

# -----------------------------

# 4. Sidebar form

# -----------------------------

if selected_page == "Dashboard":
    st.sidebar.header("📝 Student Details")

    with st.sidebar.form("student_form"):

        current_student = st.session_state.get("current_student")

        if current_student:
          default_student_name = current_student.get("name", "Student")
        else:
             default_student_name = "Student 1"

        student_name = st.text_input(
        "Student Name",
        value=default_student_name,
        disabled=True
    )


        study_hours = st.number_input(

            "Daily Study Hours",

            min_value=0.0, max_value=24.0,

            value=4.0, step=0.5

        )

        attendance = st.slider(

            "Attendance (%)", 0, 100, 75

        )

        previous_marks = st.slider(

            "Previous Marks (%)", 0, 100, 60

        )

        assignments = st.slider(

            "Assignments Completed", 0, 10, 7

        )

        sleep_hours = st.slider(

            "Daily Sleep Hours", 0, 12, 7

        )

        passing_marks = st.number_input(

            "Passing Marks",

            min_value=0, max_value=100,

            value=50, step=5

        )

        submitted = st.form_submit_button(

            "Predict & Generate Report"

        )

    # -----------------------------

    # 5. Predict and generate report

    # -----------------------------

    if submitted:

        if not student_name.strip():

            st.sidebar.error("Please enter the student name.")

            st.stop()

        student = pd.DataFrame([{

            "study_hours": study_hours,

            "attendance": attendance,

            "previous_marks": previous_marks,

            "assignments_completed": assignments,

            "sleep_hours": sleep_hours

        }])

        prediction = float(model.predict(student[features])[0])

        prediction = max(0.0, min(100.0, prediction))

        status = (

            "PASS" if prediction >= passing_marks else "FAIL"

        )

        if prediction >= 80:

            performance = "Excellent"

            advice = "Keep up the good work and maintain consistency."

        elif prediction >= 65:

            performance = "Good"

            advice = "Continue regular study and revise weak topics."

        elif prediction >= passing_marks:

            performance = "Average"

            advice = "Increase practice and review previous mistakes."

        else:

            performance = "Needs Improvement"

            advice = "Make a study plan and seek help with difficult topics."

        st.session_state["student_report"] = {

            "Student Name": student_name.strip(),
            "Student ID": st.session_state.get("current_student", {}).get("id", "N/A"),

            "Study Hours Per Day": study_hours,

            "Attendance (%)": attendance,

            "Previous Marks (%)": previous_marks,

            "Assignments Completed": assignments,

            "Sleep Hours Per Day": sleep_hours,

            "Predicted Exam Score": round(prediction, 2),

            "Passing Marks": passing_marks,

            "Result": status,

            "Performance Level": performance,

            "Recommendation": advice,

            "Report Generated": datetime.now().strftime(

                "%Y-%m-%d %H:%M:%S"

            )

        }

        # Persist only student-owned prediction reports; do not create an ADMIN-owned record.
        if st.session_state.get("user_role") == "student":
            try:
                save_report_to_database(
                    st.session_state["student_report"],
                    report_type="prediction",
                )
                st.session_state.student_reports.append(
                    st.session_state["student_report"].copy()
                )
                st.success("Report saved successfully.")
            except Exception as error:
                st.error(f"Could not save the report: {error}")
        else:
            st.session_state.student_reports.append(
                st.session_state["student_report"].copy()
            )

    # -----------------------------

    # 6. Display student report

    # -----------------------------

    report = st.session_state.get("student_report")

    if report:

        st.divider()

        st.header("📄 Student Performance Report")

        st.write(f"**Student:** {report['Student Name']}")

        st.write(f"**Report date:** {report['Report Generated']}")

        col1, col2, col3 = st.columns(3)

        col1.metric(

            "Predicted Score",

            f"{report['Predicted Exam Score']:.1f}/100"

        )

        col2.metric(

            "Result",

            report["Result"]

        )

        col3.metric(

            "Performance",

            report["Performance Level"]

        )

        if report["Result"] == "PASS":

            st.success("Predicted result: PASS")

        else:

            st.error("Predicted result: FAIL")

        st.write("**Recommendation:**", report["Recommendation"])

        report_df = pd.DataFrame([report])

        csv_data = report_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(

            "⬇️ Download Student Report (CSV)",

            data=csv_data,

            file_name="student_performance_report.csv",

            mime="text/csv"

        )

        text_report = "\n".join(

            f"{key}: {value}" for key, value in report.items()

        )

        st.download_button(

            "⬇️ Download Student Report (TXT)",

            data=text_report,

            file_name="student_performance_report.txt",

            mime="text/plain"

        )

        st.caption(

            "Predictions are experimental estimates based on a small "

            "practice dataset. They are not official exam results."

        )

    else:

        st.info(

            "Enter student details in the sidebar and click "

            "'Predict & Generate Report'."

        )

# Students can see only reports whose Student ID matches their own account.
if st.session_state.get("user_role") == "student" and selected_page == "My Reports":
    st.divider()
    st.header("📚 My Reports")
    current_student = st.session_state.get("current_student") or {}
    current_student_id = str(current_student.get("id", ""))

    # Query the database using the authenticated account's ID, never a form value.
    own_reports = get_reports_from_database(student_id=current_student_id)

    if own_reports:
        my_reports_df = pd.DataFrame(own_reports)
        st.caption(
            f"Showing {len(my_reports_df)} report(s) for your Student ID: "
            f"{current_student_id}"
        )
        st.dataframe(my_reports_df, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Download My Reports (CSV)",
            data=my_reports_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"my_reports_{current_student_id}.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info(
            "You do not have any saved reports yet. Use the sidebar to "
            "select 'Predict & Generate Report' and create a report."
        )

# -----------------------------

if st.session_state.get("user_role") == "admin" and selected_page == "Dashboard":
    if st.session_state.get("user_role") == "admin":
        # 7. Dataset performance analysis

        # -----------------------------

        st.divider()

        st.header("📊 Dataset Performance Analysis")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Records", len(df))

        col2.metric(

            "Average Exam Score",

            f"{df['exam_score'].mean():.1f}/100"

        )

        col3.metric(

            "Average Attendance",

            f"{df['attendance'].mean():.1f}%"

        )

        col4.metric(

            "Average Study Hours",

            f"{df['study_hours'].mean():.1f}"

        )

        st.subheader("Study Hours vs Exam Score")

        st.scatter_chart(

            df, x="study_hours", y="exam_score"

        )

        st.subheader("Attendance vs Exam Score")

        st.scatter_chart(

            df, x="attendance", y="exam_score"

        )

        st.subheader("Average Exam Score by Study Hours")

        average_scores = (

            df.groupby("study_hours", as_index=False)["exam_score"]

            .mean()

        )

        st.bar_chart(

            average_scores,

            x="study_hours",

            y="exam_score"

        )

        with st.expander("View Dataset Records"):

            st.dataframe(df, width="stretch")

        # Download the complete practice dataset

        st.download_button(

            "⬇️ Download Complete Dataset (CSV)",

            data=df.to_csv(index=False).encode("utf-8-sig"),

            file_name="student_dataset.csv",

            mime="text/csv",

        )

        # -----------------------------

# 8. Student Learning Gallery and article pages
# -----------------------------

if selected_page == "Student Learning Gallery":
    st.divider()
    article_page = st.session_state.get("learning_article_page")

    if article_page:
        article_content = {
            "academic": {
                "title": "🎓 Academic Journey: Small Habits, Big Results",
                "image": "student1.jpg",
                "intro": "Academic success is not built in one night. It grows from small, repeatable habits that make learning easier every day.",
                "sections": [
                    ("Create a realistic study routine", "Choose a regular time to study and split big topics into short sessions. A focused 25-minute session followed by a short break can feel much easier than trying to study for hours at once."),
                    ("Learn actively, not passively", "After reading a topic, close the book and explain the main idea in your own words. Practice questions and quick self-tests help you discover what you truly understand."),
                    ("Review and ask for help", "Review notes during the week instead of waiting until exam day. If a concept remains confusing, ask a teacher or classmate early."),
                ],
            },
            "collaborative": {
                "title": "📚 Collaborative Learning: Grow Together",
                "image": "student3.jpg",
                "intro": "Working with classmates can turn difficult topics into shared discoveries. Good teamwork combines listening, explaining and respecting different ideas.",
                "sections": [
                    ("Set one shared goal", "At the start of a group session, agree on what everyone should understand or complete. Keep the goal clear and realistic."),
                    ("Give everyone a role", "One student can summarize the topic, another can find examples, and another can prepare practice questions. Rotate roles so every member participates."),
                    ("Explain kindly and listen carefully", "When someone makes a mistake, treat it as a chance to learn. Ask questions, compare solutions and finish by summarizing what the group discovered."),
                ],
            },
            "focus": {
                "title": "💡 Focus & Progress: Make Every Session Count",
                "image": "student2.jpg",
                "intro": "Progress becomes easier to see when you protect your attention, set small goals and track the effort you make.",
                "sections": [
                    ("Reduce distractions", "Keep your phone away or silence notifications during study time. Prepare your books, water and notes before you begin so you do not keep interrupting yourself."),
                    ("Use small, measurable goals", "Instead of writing ‘study science’, write ‘solve five questions’ or ‘review two pages’. A clear task makes it easier to start and finish."),
                    ("Track progress without judging yourself", "At the end of each session, note what you completed and what needs more practice. Sleep, breaks and consistency also support concentration."),
                ],
            },
        }
        article = article_content[article_page]
        if st.button("← Back to Student Learning Gallery", key="back_to_gallery"):
            st.session_state.learning_article_page = None
            st.rerun()
        st.title(article["title"])
        article_image = IMAGE_DIR / article["image"]
        if article_image.exists():
            st.image(str(article_image), width="stretch")
        st.write(article["intro"])
        for section_title, section_text in article["sections"]:
            st.subheader(section_title)
            st.write(section_text)
        st.info("Try one tip today, then build your routine step by step. Consistency matters more than perfection.")
    else:
        st.header("🎓 Student Learning Gallery")
        st.write("Learning, collaboration and achievement — the journey to student success.")
        gallery_items = [
            ("academic", "🎓 Academic Journey", "Every achievement starts with learning.", "student1.jpg", "Open article"),
            ("collaborative", "📚 Collaborative Learning", "Learn together and grow together.", "student3.jpg", "Open article"),
            ("focus", "💡 Focus & Progress", "Small steps lead to better results.", "student2.jpg", "Open article"),
        ]
        gallery_cols = st.columns(3, gap="large")
        for gallery_col, item in zip(gallery_cols, gallery_items):
            page_key, title, caption, image_name, button_text = item
            with gallery_col:
                image_path = IMAGE_DIR / image_name
                if image_path.exists():
                    st.image(str(image_path), caption=caption, width="stretch")
                else:
                    st.info("Add " + image_name + " to the images folder to display this picture.")
                st.subheader(title)
                st.caption(caption)
                if st.button(button_text, key="open_article_" + page_key, width="stretch"):
                    st.session_state.learning_article_page = page_key
                    st.rerun()
        st.info("Student success is built through consistent effort, attendance and effective study habits.")
        st.caption("Gallery images are illustrative and do not represent the students in the practice dataset.")

# ==========================================

if st.session_state.get("user_role") == "admin":
    if selected_page == "Student Accounts":
        # ADMIN STUDENT ACCOUNT MANAGEMENT
        st.divider()
        st.header("👥 Student Account Management")
        st.caption("Registered student accounts. For forgotten passwords, use the secure reset option below; existing passwords cannot be viewed because they are hashed.")

        st.subheader("➕ Register a New Student")
        with st.form("admin_create_student_account_form"):
            new_student_id = st.text_input("Student ID", key="admin_new_student_id").strip()
            new_student_name = st.text_input("Full Name", key="admin_new_student_name").strip()
            new_student_email = st.text_input("Email (optional)", key="admin_new_student_email").strip()
            new_student_department = st.text_input("Class / Department (optional)", key="admin_new_student_department").strip()
            new_student_password = st.text_input(
                "Initial Password (minimum 6 characters)",
                type="password",
                key="admin_new_student_password",
            )
            register_student_submitted = st.form_submit_button(
                "➕ Create Student Account", width="stretch"
            )
        if register_student_submitted:
            if not new_student_id or not new_student_name:
                st.error("Student ID and full name are required.")
            elif len(new_student_password) < 6:
                st.error("Initial password must be at least 6 characters long.")
            elif get_student_account(new_student_id):
                st.error("This Student ID is already registered. Please use a unique ID.")
            else:
                try:
                    create_student_account(
                        new_student_id,
                        new_student_name,
                        new_student_password,
                        new_student_email,
                        new_student_department,
                    )
                    st.success(f"Account created for {new_student_name} ({new_student_id}).")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not create the student account: {error}")

        st.divider()
        st.subheader("📇 Registered Student Details")
        account_rows = get_all_student_accounts()
        if account_rows:
            accounts_df = pd.DataFrame(account_rows)
            search_term = st.text_input(
                "🔎 Search by Student ID or Name",
                key="admin_account_search",
                placeholder="Enter a student ID or name",
            ).strip().casefold()

            if search_term:
                filtered_accounts = accounts_df[
                    accounts_df["student_id"].astype(str).str.casefold().str.contains(search_term, na=False)
                    | accounts_df["name"].astype(str).str.casefold().str.contains(search_term, na=False)
                ]
            else:
                filtered_accounts = accounts_df

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Registered Accounts", len(accounts_df))
            metric_col2.metric("Matching Accounts", len(filtered_accounts))

            st.dataframe(
                filtered_accounts,
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "📥 Download Student Account List (CSV)",
                data=filtered_accounts.to_csv(index=False).encode("utf-8-sig"),
                file_name="student_accounts.csv",
                mime="text/csv",
                width="stretch",
            )

            st.subheader("🔑 Reset a Student Password")
            st.caption("If a student forgets their password, set a temporary new password and share it with that student privately.")
            reset_options = {
                f'{row["name"]} — {row["student_id"]}': row["student_id"]
                for row in account_rows
            }
            selected_reset_label = st.selectbox(
                "Choose student account", list(reset_options.keys()), key="admin_reset_student_select"
            )
            with st.form("admin_reset_student_password_form"):
                temporary_password = st.text_input(
                    "New temporary password (minimum 6 characters)",
                    type="password",
                    key="admin_temporary_student_password",
                )
                confirm_temporary_password = st.text_input(
                    "Confirm temporary password",
                    type="password",
                    key="admin_confirm_temporary_student_password",
                )
                reset_submitted = st.form_submit_button("Reset Student Password", width="stretch")
            if reset_submitted:
                if len(temporary_password) < 6:
                    st.error("Temporary password must be at least 6 characters long.")
                elif temporary_password != confirm_temporary_password:
                    st.error("The passwords do not match.")
                else:
                    reset_id = reset_options[selected_reset_label]
                    if reset_student_password(reset_id, temporary_password):
                        st.success(f"Password reset successfully for student ID {reset_id}. Share the new temporary password privately with the student.")
                    else:
                        st.error("Student account was not found. Refresh and try again.")

            st.subheader("🗑️ Remove a Student Account")
            st.warning(
                "The student's active account, live reports and login history will be removed. "
                "An administrator-only audit record will preserve the student details, archived reports, "
                "removal date and your remarks."
            )
            removable_accounts = [row for row in account_rows if row["student_id"] not in {"1234567", "2345678", "3456789"}]
            if removable_accounts:
                remove_options = {
                    f'{row["name"]} — {row["student_id"]}': row["student_id"]
                    for row in removable_accounts
                }
                selected_remove_label = st.selectbox(
                    "Choose student to remove", list(remove_options.keys()), key="admin_remove_student_select"
                )
                removal_remarks = st.text_area(
                    "Admin remarks / reason for removal (required)",
                    key="admin_remove_student_remarks",
                    placeholder="Example: Student transferred to another institution after parent request.",
                )
                confirm_remove = st.checkbox(
                    "I confirm this removal and understand it will be recorded in Removed Students.",
                    key=f"admin_confirm_remove_student_{remove_options[selected_remove_label]}",
                )
                if st.button(
                    "🗑️ Remove Selected Student", 
                    disabled=(not confirm_remove or not removal_remarks.strip()),
                    key="admin_remove_student_button",
                ):
                    removed_id = remove_options[selected_remove_label]
                    admin_name = (st.session_state.get("current_student") or {}).get("name", "Administrator")
                    try:
                        if delete_student_account(removed_id, removal_remarks, admin_name):
                            st.success(f"Student {removed_id} removed. Reason and student history are saved in 'Removed Students'.")
                            st.rerun()
                        else:
                            st.error("Student account was not found. Refresh and try again.")
                    except Exception as error:
                        st.error(f"Could not remove student: {error}")
            else:
                st.info("No removable student accounts found. Reserved demo accounts are protected from deletion.")
        else:
            st.info("No student accounts are registered yet.")

    if selected_page == "Removed Students":
        st.divider()
        st.header("🗃️ Removed Students & Audit History")
        st.caption(
            "Administrator-only history. Removal remarks and archived report summaries remain available "
            "even after the active account has been deleted."
        )
        removed_rows = get_removed_students()
        if removed_rows:
            removed_df = pd.DataFrame([
                {key: value for key, value in row.items() if key != "archived_reports"}
                for row in removed_rows
            ])
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Removal Records", len(removed_df))
            metric2.metric("Archived Reports", int(removed_df["reports_count"].sum()))
            metric3.metric("Previous Logins", int(removed_df["login_count"].sum()))
            st.dataframe(removed_df, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download Removed Students Audit (CSV)",
                data=removed_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="removed_students_audit.csv",
                mime="text/csv",
                width="stretch",
            )
            st.subheader("🔎 View Archived Details")
            archived_options = {
                f'{row["student_name"]} — {row["student_id"]} — {row["removed_at"]} (Record #{row["removal_id"]})': row
                for row in removed_rows
            }
            selected_archive_label = st.selectbox(
                "Choose a removed student record", list(archived_options.keys()), key="admin_removed_record_select"
            )
            selected_archive = archived_options[selected_archive_label]
            details_col1, details_col2 = st.columns(2)
            details_col1.write(f"**Student ID:** {selected_archive['student_id']}")
            details_col1.write(f"**Name:** {selected_archive['student_name']}")
            details_col1.write(f"**Email:** {selected_archive['email'] or 'Not provided'}")
            details_col1.write(f"**Department:** {selected_archive['department'] or 'Not provided'}")
            details_col2.write(f"**Removed at:** {selected_archive['removed_at']}")
            details_col2.write(f"**Removed by:** {selected_archive['removed_by']}")
            details_col2.write(f"**Reports archived:** {selected_archive['reports_count']}")
            details_col2.write(f"**Previous logins:** {selected_archive['login_count']}")
            st.info("**Removal remarks:** " + selected_archive["remarks"])
            st.write(f"**Original account created:** {selected_archive['account_created_at'] or 'Unknown'}")
            archived_reports = selected_archive.get("archived_reports", [])
            if archived_reports:
                st.subheader("📄 Archived Student Reports")
                archived_df = pd.DataFrame(archived_reports)
                st.dataframe(archived_df, width="stretch", hide_index=True)
                st.download_button(
                    "⬇️ Download Archived Reports (CSV)",
                    data=archived_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"archived_reports_{selected_archive['student_id']}.csv",
                    mime="text/csv",
                    key=f"download_archived_reports_{selected_archive['removal_id']}",
                )
            else:
                st.info("No saved reports existed for this student at the time of removal.")
        else:
            st.info("Abhi koi student remove nahi hua. Jab admin remarks ke saath student remove karega, record yahan nazar aayega.")

    if selected_page == "Login History":
        st.divider()
        st.header("🔐 Student Login History")
        st.caption("Successful sign-ins are recorded here. Passwords are never saved in login history.")
        login_rows = get_student_login_history()
        if login_rows:
            login_df = pd.DataFrame(login_rows)
            st.metric("Recorded Student Logins", len(login_df))
            st.dataframe(login_df, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download Login History (CSV)",
                data=login_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="student_login_history.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.info("No successful student logins have been recorded yet. New logins will appear here.")

    if selected_page == "Student Reports":
        # STUDENT REPORTS AND PERFORMANCE ANALYSIS

        # ==========================================

        st.divider()

        st.header("📋 Student Reports")

        st.write("Add student details and generate an academic report.")

        with st.form("student_report_form"):

            st.subheader("👨‍🎓 Student Information")

            student_name = st.text_input("Student Name")

            student_id = st.text_input("Student ID")

            st.subheader("📚 Academic Details")

            col1, col2 = st.columns(2)

            with col1:

                study_hours = st.number_input(

                    "Study Hours per Day",

                    min_value=0.0,

                    max_value=24.0,

                    value=2.0

                )

                attendance = st.number_input(

                    "Attendance (%)",

                    min_value=0,

                    max_value=100,

                    value=75

                )

                previous_marks = st.number_input(

                    "Previous Marks",

                    min_value=0,

                    max_value=100,

                    value=50

                )

            with col2:

                assignments_completed = st.number_input(

                    "Assignments Completed",

                    min_value=0,

                    value=5

                )

                sleep_hours = st.number_input(

                    "Sleep Hours",

                    min_value=0.0,

                    max_value=24.0,

                    value=7.0

                )

                exam_score = st.number_input(

                    "Exam Score (out of 100)",

                    min_value=0,

                    max_value=100,

                    value=50

                )

            submitted = st.form_submit_button(

                "➕ Generate Student Report",

                width="stretch"

            )

        if submitted:

                if not student_name.strip():

                    st.error("Please enter Student Name.")

                elif not student_id.strip():

                    st.error("Student Not Found: Please enter Student ID.")

                elif not student_id.strip().isdigit():

                    st.error(

                        "Student Not Found: ID must contain numeric digits only."

                    )

                elif len(student_id.strip()) < 7:

                    st.error(

                        "Student Not Found: Student ID must be at least 7 digits."

                    )

                elif any(

                    report["Student ID"] == student_id.strip()

                    for report in st.session_state.student_reports

                ):

                    st.error("This Student ID already exists.")

                else:

                    percentage = float(exam_score)

                    if exam_score >= 80:

                        grade = "A"

                    elif exam_score >= 70:

                        grade = "B"

                    elif exam_score >= 60:

                        grade = "C"

                    elif exam_score >= 40:

                        grade = "D"

                    else:

                        grade = "F"

                    result = "Pass" if exam_score >= 40 else "Fail"

                    if exam_score >= 80:

                        performance = "Excellent"

                    elif exam_score >= 60:

                        performance = "Good"

                    elif exam_score >= 40:

                        performance = "Needs Improvement"

                    else:

                        performance = "At Risk"

                    student_report = {

                        "Student ID": student_id.strip(),

                        "Student Name": student_name.strip(),

                        "Study Hours": study_hours,

                        "Attendance (%)": attendance,

                        "Previous Marks": previous_marks,

                        "Assignments Completed": assignments_completed,

                        "Sleep Hours": sleep_hours,

                        "Exam Score": exam_score,

                        "Percentage": percentage,

                        "Grade": grade,

                        "Result": result,

                        "Performance": performance

                    }

                    try:
                        save_report_to_database(
                            student_report,
                            report_type="admin_manual",
                        )
                        st.session_state.student_reports.append(student_report)
                        st.success(
                            f"Report generated and saved for {student_name}!"
                        )
                    except Exception as error:
                        st.error(f"Could not save the report: {error}")

        # Display saved student reports

        all_saved_reports = get_reports_from_database()

        if all_saved_reports:

            st.divider()

            st.subheader("📊 Student Performance Analysis")

            reports_df = pd.DataFrame(all_saved_reports)

            total_students = len(reports_df)

            passed_students = int(
                reports_df["Result"].astype(str).str.strip().str.lower().eq("pass").sum()
            )

            failed_students = total_students - passed_students

            pass_percentage = (passed_students / total_students) * 100

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Reports", total_students)

            col2.metric("Passed", passed_students)

            col3.metric("Failed", failed_students)

            col4.metric("Pass Rate", f"{pass_percentage:.1f}%")

            st.subheader("📝 All Student Reports")

            st.dataframe(reports_df, width="stretch")

            st.download_button(

                label="📥 Download Student Reports CSV",

                data=reports_df.to_csv(index=False).encode("utf-8-sig"),

                file_name="student_performance_reports.csv",

                mime="text/csv",

                width="stretch"

            )

            st.subheader("📈 Pass / Fail Chart")

            result_counts = (

                reports_df["Result"]

                .value_counts()

                .reindex(["Pass", "Fail"], fill_value=0)

            )

            st.bar_chart(result_counts)


        st.divider()
        st.subheader("🗃️ Removed Students — Removal Reasons")
        removed_report_rows = get_removed_students()
        if removed_report_rows:
            removed_report_df = pd.DataFrame([
                {
                    "Student ID": row["student_id"],
                    "Student Name": row["student_name"],
                    "Removed At": row["removed_at"],
                    "Removed By": row["removed_by"],
                    "Reason / Admin Remarks": row["remarks"],
                    "Archived Reports": row["reports_count"],
                    "Previous Logins": row["login_count"],
                }
                for row in removed_report_rows
            ])
            st.dataframe(removed_report_df, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download Removal Reasons for Reports (CSV)",
                data=removed_report_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="student_removal_reasons.csv",
                mime="text/csv",
                key="download_student_removal_reasons_csv",
            )
        else:
            st.info("No removed-student records are available yet.")
