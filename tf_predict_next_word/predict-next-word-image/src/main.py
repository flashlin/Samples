from flask import Flask

# import pyodbc
# cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=your_database;UID=your_username;PWD=your_password')

# # 建立 cursor
# cursor = cnxn.cursor()

# # 執行 SQL Query
# cursor.execute("SELECT * FROM your_table")

# # 取得查詢結果
# rows = cursor.fetchall()

# # 關閉 cursor 和連接
# cursor.close()
# cnxn.close()




app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def hello_world():
    return "Hello, World!"