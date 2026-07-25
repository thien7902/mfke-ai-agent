"""MongoDB connection helper."""
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from src.bot.utils.config import config


class MongoDB:
    """MongoDB connection manager."""

    _client: MongoClient = None
    _database: Database = None

    @classmethod
    def get_client(cls) -> MongoClient:
        """Get or create MongoDB client."""
        if cls._client is None:
            cls._client = MongoClient(config.mongodb_uri)
        return cls._client

    @classmethod
    def get_database(cls) -> Database:
        """Get or create database reference."""
        if cls._database is None:
            cls._database = cls.get_client()[config.mongodb_database]
        return cls._database

    @classmethod
    def get_collection(cls, name: str) -> Collection:
        """Get a collection by name."""
        return cls.get_database()[name]

    @classmethod
    def close(cls):
        """Close the MongoDB connection."""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._database = None


def get_mongodb() -> Database:
    """Dependency injection helper for database."""
    return MongoDB.get_database()