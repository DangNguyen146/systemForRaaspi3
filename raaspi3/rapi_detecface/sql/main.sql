DROP TABLE IF EXISTS useropendoor;
CREATE TABLE useropendoor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fullname TEXT NOT NULL,
    age INTEGER,
    email TEXT,
    create_time datetime not null default(current_timestamp)
);