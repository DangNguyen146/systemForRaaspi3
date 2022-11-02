import sqlite3

connection = sqlite3.connect('./static/sql/useropendoor.db')


with open('./static/sql/main.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

connection.commit()
connection.close()