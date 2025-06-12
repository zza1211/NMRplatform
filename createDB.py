import sqlite3

conn = sqlite3.connect('metanaly.db')
print("Opened database successfully")

conn.execute('CREATE TABLE file_info (id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, object TEXT, context TEXT, sample TEXT, gp TEXT, date DATE, purpose TEXT)')
print("Table created successfully")
conn.close()
