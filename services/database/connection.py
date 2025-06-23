import sqlite3

def register_animal(study_id, specie, sex, age, weight):
    conn = sqlite3.connect("animal-db.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO animal (study_id, specie, sex, age, weight) VALUES (?, ?, ?, ?, ?) ", (study_id, specie, sex, age, weight))
    conn.commit()
    conn.close()

def list_animals():
    conn = sqlite3.connect("animal-db.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM animal")
    animals = cursor.fetchall()
    conn.close()
    return animals