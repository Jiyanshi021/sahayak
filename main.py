
# import os
# from dotenv import load_dotenv
# import re
# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_groq import ChatGroq
# from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
# from langsmith import traceable
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate

# # 1) Load .env and print for debug
# load_dotenv()
# print("LANGSMITH_TRACING:", os.getenv("LANGSMITH_TRACING"))
# print("LANGSMITH_API_KEY:", os.getenv("LANGSMITH_API_KEY"))
# print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
# os.environ["GROQ_API_KEY"]   = os.getenv("GROQ_API_KEY", "")

# # 2) Connect to your PostgreSQL
# db_user = "neondb_owner"
# db_password = "npg_1bkheTAa7dcL"
# db_host = "ep-dawn-morning-a1w3xy1h-pooler.ap-southeast-1.aws.neon.tech"
# db_name = "neondb"
# db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}"
# db = SQLDatabase.from_uri(db_uri)

# print("Dialect:", db.dialect)
# print("Usable tables:", db.get_usable_table_names())

# @traceable
# def main():
#     question = "How many internships are there with stipend more than 1000"
#     llm = ChatGroq(model="llama3-8b-8192")

#     # 3) Generate the raw SQL response
#     generate_query = create_sql_query_chain(llm, db)
#     raw = generate_query.invoke({"question": question})
#     print("Raw response:\n", raw)

#     # 4) Pull out just the SQL text after "SQLQuery:"
#     if "SQLQuery:" in raw:
#         _, after = raw.split("SQLQuery:", 1)
#         sql_query = after.strip()
#     else:
#         sql_query = raw.strip()

#     # 5) Remove markdown fences if present
#     if sql_query.startswith("```"):
#         sql_query = sql_query.split("\n", 1)[1]
#     if sql_query.endswith("```"):
#         sql_query = sql_query.rsplit("```", 1)[0]

#     # 6) Strip any leading prose up to the first SELECT
#     idx = sql_query.upper().find("SELECT")
#     if idx != -1:
#         sql_query = sql_query[idx:].strip()

#     print("Clean SQL query:\n", sql_query)

#     # 7) Execute it
#     exec_tool = QuerySQLDatabaseTool(db=db)
#     result = exec_tool.invoke(sql_query)  # e.g. [(62,)]
#     print("Query result raw:", result)

#     # 8) Rephrase the answer for the **same** question
#     answer_prompt = PromptTemplate.from_template(
#         """Given the following user question, SQL query, and SQL result, answer succinctly.

# Question: {question}
# SQL Query: {query}
# SQL Result: {result}
# Answer: """
#     )
#     rephraser = answer_prompt | llm | StrOutputParser()
#     answer = rephraser.invoke({
#         "question": question,
#         "query": sql_query,
#         "result": result
#     })
#     print("Final answer:", answer)

# if __name__ == "__main__":
#     main()





# import os
# import pandas as pd
# import psycopg2
# from flask import Flask, render_template, request, jsonify
# from dotenv import load_dotenv
# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_groq import ChatGroq
# from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
# from langsmith import traceable
# from sqlalchemy import create_engine

# # Load environment variables
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# # Flask app initialization
# app = Flask(__name__)

# # Connect to DB
# db_user = "neondb_owner"
# db_password = "npg_1bkheTAa7dcL"
# db_host = "ep-dawn-morning-a1w3xy1h-pooler.ap-southeast-1.aws.neon.tech"
# db_name = "neondb"
# db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}"
# db = SQLDatabase.from_uri(db_uri)

# # SQLAlchemy engine for pandas
# engine = create_engine(db_uri)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @traceable
# @app.route('/ask', methods=['POST'])
# def ask():
#     question = request.form['question']
#     llm = ChatGroq(model="llama3-8b-8192")

#     try:
#         # Step 1: Generate SQL query
#         generate_query = create_sql_query_chain(llm, db)
#         raw = generate_query.invoke({"question": question})

#         # Step 2: Extract SQL query from raw result
#         if "SQLQuery:" in raw:
#             _, after = raw.split("SQLQuery:", 1)
#             sql_query = after.strip()
#         else:
#             sql_query = raw.strip()

#         if sql_query.startswith("```"):
#             sql_query = sql_query.split("\n", 1)[1]
#         if sql_query.endswith("```"):
#             sql_query = sql_query.rsplit("```", 1)[0]

#         if "Question:" in sql_query:
#             idx = sql_query.upper().find("SELECT")
#             sql_query = sql_query[idx:]

#         # Step 3: Execute SQL query using LangChain
#         exec_tool = QuerySQLDatabaseTool(db=db)
#         result = exec_tool.invoke(sql_query)

#         # Step 4: Convert SQL result to natural language
#         prompt = f"""Given the original question: "{question}" and the result of SQL query: {result}, 
# provide a concise natural language answer."""
#         answer = str(llm.invoke(prompt))  # ✅ Fix for JSON serialization

#         # Step 5: Convert result to HTML table using pandas
#         df_result = pd.read_sql_query(sql_query, engine)
#         table_html = df_result.to_html(classes="table table-striped", index=False)
        
#         ans= "This is the best suitable answer"


#         # Step 6: Return both
#         return jsonify({'answer': ans, 'table': table_html})
    
#     except Exception as e:
#         return jsonify({'answer': f"Error: {str(e)}", 'table': ''})

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
import bcrypt
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# Initialize app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup LangChain SQL Query Generator
db = SQLDatabase.from_uri("sqlite:///db.db")
llm = ChatGroq(model="llama3-8b-8192")
write_query = create_sql_query_chain(llm, db)

def get_database_connection():
    return sqlite3.connect("db.db")

@app.route('/', methods=['GET'])
def home():
    return "Flask server is running. POST to /ask to query."

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "No question provided.", "table": ""}), 400

    try:
        generated_query = write_query.invoke({"question": question})
        print("🔍 LLM Raw Output:", generated_query)

        if "SQLQuery:" not in generated_query:
            return jsonify({"answer": "Could not generate SQL query.", "table": ""}), 400

        sql_query = generated_query.split("SQLQuery: ")[1].strip()
        print("🧠 Extracted SQL:", sql_query)

        conn = get_database_connection()
        try:
            df_result = pd.read_sql_query(sql_query, conn)
        except Exception as sql_err:
            print("⚠️ SQL Error:", sql_err)
            return jsonify({"answer": f"SQL execution failed: {str(sql_err)}", "table": ""}), 500
        finally:
            conn.close()

        if df_result.empty:
            return jsonify({"answer": "No results found.", "table": ""})

        # 🛠️ Split 'titleorganization' into two columns
        if 'titleorganization' in df_result.columns:
            split_cols = df_result['titleorganization'].astype(str).str.split(r'\n| {2,}', expand=True)
            split_cols.columns = ['title', 'organization']
            df_result.drop(columns=['titleorganization'], inplace=True)
            df_result = pd.concat([split_cols, df_result], axis=1)

        # 🌐 Make 'link' column clickable
        if 'link' in df_result.columns:
            df_result['link'] = df_result['link'].apply(
                lambda url: f'<a href="{url}" target="_blank" class="text-blue-400 underline break-words">{url}</a>'
            )

        # 🧾 Convert DataFrame to HTML with styling
               # 🧾 Convert DataFrame to HTML with styling
        table_html = df_result.to_html(
            index=False,
            escape=False,
            border=0,
            classes="",
        )

        table_html = table_html.replace(
            "<table",
            '<table style="width:100%; border-collapse: collapse; color: white; background-color: #1f2937;"'
        ).replace(
            "<th",
            '<th style="border: 1px solid #4b5563; padding: 8px; background-color: #374151;"'
        ).replace(
            "<td",
            '<td style="border: 1px solid #4b5563; padding: 8px;"'
        )

        return jsonify({'answer': "This is the best suitable answer:", 'table': table_html})

    except Exception as e:
        print("❌ Exception:", e)
        return jsonify({'answer': f"Error: {str(e)}", 'table': ''})


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, 
                name TEXT, 
                email TEXT UNIQUE, 
                password TEXT
            )
        """)
        cursor.execute("SELECT COUNT(*) FROM users WHERE email=?", (email,))
        if cursor.fetchone()[0] > 0:
            return jsonify({"error": "Email already registered"}), 400

        cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_password))
        conn.commit()
        conn.close()

        return jsonify({"message": "Registration successful"}), 200

    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[3]):
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    except Exception as e:
        return jsonify({"error": f"Login error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
