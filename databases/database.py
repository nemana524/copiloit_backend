from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from config.config import DATABASE_URL
import os

# Create Base class for models
Base = declarative_base()

# Define models - no engine connection yet
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    repositories = relationship("Repository", back_populates="owner")

class Repository(Base):
    __tablename__ = "repositories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    phone_number = Column(String, ForeignKey("users.phone_number"))
    owner = relationship("User", back_populates="repositories")
    files = relationship("FileRecord", back_populates="repository")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FileRecord(Base):
    __tablename__ = "file_records"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    original_filename = Column(String)
    file_size = Column(Integer)
    mime_type = Column(String)
    phone_number = Column(String, ForeignKey("users.phone_number"))
    repository_id = Column(Integer, ForeignKey("repositories.id"))
    repository = relationship("Repository", back_populates="files")
    storage_path = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Lazy-loaded engine and session factory
_engine = None
_SessionLocal = None

def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
    return _engine

def get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal

def get_db():
    """Get a database session. Used as a dependency in FastAPI."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_db():
    """Initialize database tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

# Only create tables if we're not in production mode
# This allows Render to set up the proper environment variables first
if os.environ.get('ENV') != 'production':
    initialize_db()