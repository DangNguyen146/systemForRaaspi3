import sqlite3

connection = sqlite3.connect('./sql/useropendoor.db')


with open('./sql/main.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

connection.commit()
connection.close()