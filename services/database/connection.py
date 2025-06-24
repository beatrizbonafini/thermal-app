from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def init_db():
    engine = create_engine('sqlite:///animal-db.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    return session
