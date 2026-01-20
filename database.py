"""
MongoDB Connection Module
Handles all database connections and configurations
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import streamlit as st
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBConnection:
    """MongoDB Connection Manager"""
    
    def __init__(self):
        self.client = None
        self.database = None
        self.connection_string = self._get_connection_string()
        
    def _get_connection_string(self):
        """Get MongoDB connection string from environment variables or Streamlit secrets"""
        try:
            # Try to get from Streamlit secrets first (recommended for deployment)
            if hasattr(st, 'secrets') and 'mongodb' in st.secrets:
                return st.secrets['mongodb']['connection_string']
            
            # Fallback to environment variables
            mongodb_uri = os.getenv('MONGODB_URI')
            if mongodb_uri:
                return mongodb_uri
                
            # Default local MongoDB connection
            host = os.getenv('MONGODB_HOST', 'localhost')
            port = os.getenv('MONGODB_PORT', '27017')
            username = os.getenv('MONGODB_USERNAME')
            password = os.getenv('MONGODB_PASSWORD')
            database_name = os.getenv('MONGODB_DATABASE', 'virus_prediction')
            
            if username and password:
                return f"mongodb://{username}:{password}@{host}:{port}/{database_name}"
            else:
                return f"mongodb://{host}:{port}/{database_name}"
                
        except Exception as e:
            logger.warning(f"Could not load MongoDB configuration: {e}")
            return "mongodb://localhost:27017/virus_prediction"  # Default fallback
    
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            if not self.client:
                self.client = MongoClient(
                    self.connection_string,
                    serverSelectionTimeoutMS=5000,  # 5 seconds timeout
                    connectTimeoutMS=10000,         # 10 seconds timeout
                    maxPoolSize=10,                 # Maximum connection pool size
                    retryWrites=True
                )
                
                # Test the connection
                self.client.admin.command('ping')
                
                # Get database name from connection string or use default
                db_name = os.getenv('MONGODB_DATABASE', 'virus_prediction')
                self.database = self.client[db_name]
                
                logger.info("Successfully connected to MongoDB")
                return True
                
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return False
    
    def disconnect(self):
        """Close MongoDB connection"""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.database = None
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
    
    def get_database(self):
        """Get database instance"""
        if not self.database:
            if self.connect():
                return self.database
            else:
                return None
        return self.database
    
    def test_connection(self):
        """Test MongoDB connection and return status"""
        try:
            if self.connect():
                # Test with a simple operation
                db = self.get_database()
                if db:
                    db.list_collection_names()
                    return {
                        'status': 'success',
                        'message': 'Successfully connected to MongoDB',
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'status': 'error',
                'message': 'Failed to connect to MongoDB',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Connection test failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

# Global connection instance
mongo_connection = MongoDBConnection()

def get_db():
    """Get database instance - use this function in your app"""
    return mongo_connection.get_database()

def test_db_connection():
    """Test database connection - use this function to check status"""
    return mongo_connection.test_connection()

def close_db_connection():
    """Close database connection - call this when app shuts down"""
    mongo_connection.disconnect()