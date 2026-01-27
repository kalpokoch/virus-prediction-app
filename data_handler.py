"""
Data Handler Module
Handles all database operations and data persistence
"""
from datetime import datetime
import pandas as pd
import logging
from database import get_db, test_db_connection
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class DataHandler:
    """Handles all database operations for the virus prediction app"""
    
    def __init__(self):
        self.db = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database connection"""
        try:
            self.db = get_db()
            if self.db is not None:
                # Create indexes for better performance
                self._create_indexes()
                logger.info("DataHandler initialized successfully")
            else:
                logger.warning("Failed to initialize database connection")
        except Exception as e:
            logger.error(f"Error initializing DataHandler: {e}")
    
    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            if self.db is None:
                return
                
            # Create indexes on frequently queried fields
            collections = {
                'predictions': [
                    ('timestamp', -1),
                    ('patient_id', 1),
                    ('predicted_virus', 1),
                    ('validation.validated', 1),
                    ('validation.actual_virus_name', 1)
                ],
                'patients': [
                    ('patient_id', 1),
                    ('created_at', -1)
                ],
                'usage_stats': [
                    ('date', -1),
                    ('prediction_count', 1)
                ]
            }
            
            for collection_name, indexes in collections.items():
                collection = self.db[collection_name]
                for index_fields in indexes:
                    try:
                        collection.create_index([index_fields])
                    except Exception as e:
                        logger.warning(f"Index creation warning for {collection_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def save_prediction(self, 
                       patient_data: Dict, 
                       prediction_result: Dict,
                       model_info: Dict = None) -> Optional[str]:
        """
        Save prediction result to database
        
        Args:
            patient_data: Patient information and symptoms
            prediction_result: Model prediction results
            model_info: Model version and metadata
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            collection = self.db['predictions']
            
            # Prepare document
            document = {
                'timestamp': datetime.utcnow(),
                'patient_data': patient_data,
                'prediction_result': prediction_result,
                'model_info': model_info or {},
                'app_version': '1.0'  # You can make this dynamic
            }
            
            # Insert document
            result = collection.insert_one(document)
            
            # Update usage statistics
            self._update_usage_stats()
            
            logger.info(f"Prediction saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return None
    
    def save_patient(self, patient_data: Dict) -> Optional[str]:
        """
        Save patient information to database
        
        Args:
            patient_data: Patient demographic and clinical information
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            collection = self.db['patients']
            
            # Add metadata
            document = {
                **patient_data,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            
            result = collection.insert_one(document)
            logger.info(f"Patient saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving patient: {e}")
            return None
    
    def get_prediction_history(self, 
                              limit: int = 100,
                              patient_id: str = None) -> List[Dict]:
        """
        Retrieve prediction history
        
        Args:
            limit: Maximum number of records to return
            patient_id: Filter by specific patient ID
            
        Returns:
            List of prediction records
        """
        try:
            if self.db is None:
                return []
            
            collection = self.db['predictions']
            
            # Build query
            query = {}
            if patient_id:
                query['patient_data.patient_id'] = patient_id
            
            # Get records
            cursor = collection.find(query).sort('timestamp', -1).limit(limit)
            records = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for record in records:
                record['_id'] = str(record['_id'])
                
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving prediction history: {e}")
            return []
    
    def get_usage_statistics(self) -> Dict:
        """
        Get usage statistics
        
        Returns:
            Dictionary with usage statistics
        """
        try:
            if self.db is None:
                return {}
            
            predictions_collection = self.db['predictions']
            
            # Get total predictions
            total_predictions = predictions_collection.count_documents({})
            
            # Get predictions by virus type
            pipeline = [
                {
                    '$group': {
                        '_id': '$prediction_result.predicted_virus',
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'count': -1}}
            ]
            
            virus_stats = list(predictions_collection.aggregate(pipeline))
            
            # Get predictions by date (last 30 days)
            from datetime import timedelta
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            daily_pipeline = [
                {
                    '$match': {
                        'timestamp': {'$gte': thirty_days_ago}
                    }
                },
                {
                    '$group': {
                        '_id': {
                            '$dateToString': {
                                'format': '%Y-%m-%d',
                                'date': '$timestamp'
                            }
                        },
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'_id': 1}}
            ]
            
            daily_stats = list(predictions_collection.aggregate(daily_pipeline))
            
            return {
                'total_predictions': total_predictions,
                'virus_distribution': virus_stats,
                'daily_predictions': daily_stats,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting usage statistics: {e}")
            return {}
    
    def _update_usage_stats(self):
        """Update daily usage statistics"""
        try:
            if self.db is None:
                return
            
            collection = self.db['usage_stats']
            today = datetime.utcnow().date().isoformat()
            
            # Update or create today's stats
            collection.update_one(
                {'date': today},
                {
                    '$inc': {'prediction_count': 1},
                    '$set': {'last_updated': datetime.utcnow()}
                },
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error updating usage stats: {e}")
    
    def save_validation(self, validation_data: Dict) -> Optional[str]:
        """
        Save medical validation data within the prediction document
        
        Args:
            validation_data: Validation information including actual diagnosis
            
        Returns:
            Prediction document ID if successful, None otherwise
        """
        try:
            if self.db is None:
                logger.error("Database not initialized")
                return None
            
            collection = self.db['predictions']
            prediction_id = validation_data.get('prediction_id')
            
            if not prediction_id:
                logger.error("No prediction_id provided in validation data")
                return None
            
            # Convert string ID to ObjectId for MongoDB
            from bson import ObjectId
            try:
                object_id = ObjectId(prediction_id)
            except Exception as e:
                logger.error(f"Invalid prediction_id format: {e}")
                return None
            
            # Prepare validation document (remove prediction_id as it's not needed in the validation field)
            validation_doc = {k: v for k, v in validation_data.items() if k != 'prediction_id'}
            validation_doc['validated_at'] = datetime.utcnow()
            validation_doc['validated'] = True
            
            # Update the prediction document with validation data
            result = collection.update_one(
                {'_id': object_id},
                {
                    '$set': {'validation': validation_doc},
                    '$currentDate': {'last_updated': True}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Validation added to prediction ID: {prediction_id}")
                return prediction_id
            else:
                logger.warning(f"No prediction found with ID: {prediction_id}")
                return None
            
        except Exception as e:
            logger.error(f"Error saving validation: {e}")
            return None
    
    def get_validation_stats(self) -> Dict:
        """
        Get validation statistics for data collection and analysis
        Note: This is for research/improvement purposes, not system accuracy calculation
        
        Returns:
            Dictionary containing validation collection statistics
        """
        try:
            if self.db is None:
                return {'status': 'error', 'message': 'Database not initialized'}
            
            collection = self.db['predictions']
            
            # Count total predictions with validation data
            total_validations = collection.count_documents({'validation.validated': True})
            
            # Get validation distribution by actual virus
            pipeline = [
                {
                    '$match': {'validation.validated': True}
                },
                {
                    '$group': {
                        '_id': '$validation.actual_virus_name',
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'count': -1}}
            ]
            
            validation_distribution = list(collection.aggregate(pipeline))
            
            # Get validation confidence distribution
            confidence_pipeline = [
                {
                    '$match': {'validation.validated': True}
                },
                {
                    '$group': {
                        '_id': '$validation.confidence_level',
                        'count': {'$sum': 1}
                    }
                }
            ]
            
            confidence_stats = list(collection.aggregate(confidence_pipeline))
            
            return {
                'status': 'success',
                'total_validations': total_validations,
                'validation_distribution': validation_distribution,
                'confidence_distribution': confidence_stats,
                'last_updated': datetime.utcnow().isoformat(),
                'note': 'This data is for research and model improvement purposes'
            }
            
        except Exception as e:
            logger.error(f"Error getting validation stats: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def health_check(self) -> Dict:
        """
        Perform health check on database connection and operations
        
        Returns:
            Health check results
        """
        try:
            # Check if we have a database instance (connection already established)
            if self.db is None:
                return {
                    'status': 'error',
                    'message': 'Database instance not available',
                    'details': {
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
            
            # Test basic operations using existing connection
            try:
                # Simple test - count documents in predictions collection
                predictions_count = self.db['predictions'].count_documents({})
                
                return {
                    'status': 'healthy',
                    'message': 'All database operations working',
                    'details': {
                        'connection': 'OK',
                        'total_predictions': predictions_count,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
                
            except Exception as op_error:
                return {
                    'status': 'error',
                    'message': f'Database operations failed: {str(op_error)}',
                    'details': {
                        'error': str(op_error),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'details': {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }

# Global data handler instance
data_handler = DataHandler()

# Convenience functions for use in app.py
def save_prediction_to_db(patient_data: Dict, prediction_result: Dict, model_info: Dict = None) -> Optional[str]:
    """Save prediction to database"""
    return data_handler.save_prediction(patient_data, prediction_result, model_info)

def save_validation_to_db(validation_data: Dict) -> Optional[str]:
    """Save validation to database"""
    return data_handler.save_validation(validation_data)

def get_db_health() -> Dict:
    """Get database health status"""
    return data_handler.health_check()

def get_prediction_stats() -> Dict:
    """Get prediction usage statistics"""
    return data_handler.get_usage_statistics()

def get_validation_stats() -> Dict:
    """Get validation statistics"""
    return data_handler.get_validation_stats()