# mongodb_handler.py
"""
MongoDB integration module for Cash Reconciliation application
"""

import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


class MongoDBHandler:
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="test_anp"):
        """
        Initialize MongoDB connection

        Args:
            connection_string: MongoDB connection URI
            db_name: Name of the database to use
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False
        self._connect()

    def _connect(self):
        """Establish MongoDB connection"""
        try:
            # directConnection=True only works for single-node localhost setups.
            # Atlas / Fly.io / other managed services use replica sets -- auto-detect.
            _is_localhost = (
                "localhost" in self.connection_string
                or "127.0.0.1" in self.connection_string
            )
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=8000,
                socketTimeoutMS=8000,
                directConnection=_is_localhost,
            )

            # Test connection
            self.client.admin.command("ping")

            self.db = self.client[self.db_name]
            self.connected = True
            print(f"[MongoDB] Connected to database: {self.db_name}")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.connected = False
            print(f"[MongoDB] Connection failed: {str(e)}")
            print("  Application will use file-based storage as fallback")

        except Exception as e:
            self.connected = False
            print(f"[MongoDB] Unexpected error: {str(e)}")
            print("  Application will use file-based storage as fallback")

    def is_connected(self):
        """Check if MongoDB is connected"""
        return self.connected

    # --------------------------------------------------------------------------------------
    # Session Data Operations (rec.pkl, reconciliation data)
    # --------------------------------------------------------------------------------------

    def save_session_rec(self, session_id, df, metadata=None):
        """Save reconciliation DataFrame for a session."""
        if not self.connected:
            return False
        try:
            collection = self.db['session_rec']
            records = df.to_dict('records')
            doc = {
                'session_id': session_id,
                'data': records,
                'columns': list(df.columns),
                'metadata': metadata or {},
                'updated_at': datetime.now()
            }
            collection.update_one(
                {'session_id': session_id},
                {'$set': doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving session rec to MongoDB: {str(e)}")
            return False

    def load_session_rec(self, session_id):
        """Load reconciliation DataFrame for a session. Returns DataFrame or None."""
        if not self.connected:
            return None
        try:
            collection = self.db['session_rec']
            doc = collection.find_one({'session_id': session_id})
            if doc and 'data' in doc:
                return pd.DataFrame(doc['data'], columns=doc.get('columns', []))
            return None
        except Exception as e:
            print(f"Error loading session rec from MongoDB: {str(e)}")
            return None

    # --------------------------------------------------------------------------------------
    # Carry Forward Operations (carry_unmatched.pkl)
    # --------------------------------------------------------------------------------------

    def save_carry_forward(self, account, df):
        """Save carry-forward unmatched data for an account."""
        if not self.connected:
            return False
        try:
            collection = self.db['carry_forward']
            records = df.to_dict('records')
            doc = {
                'account': account,
                'data': records,
                'columns': list(df.columns),
                'updated_at': datetime.now()
            }
            collection.update_one(
                {'account': account},
                {'$set': doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving carry forward to MongoDB: {str(e)}")
            return False

    def load_carry_forward(self, account):
        """Load carry-forward unmatched data for an account. Returns DataFrame or None."""
        if not self.connected:
            return None
        try:
            collection = self.db['carry_forward']
            doc = collection.find_one({'account': account})
            if doc and 'data' in doc:
                return pd.DataFrame(doc['data'], columns=doc.get('columns', []))
            return None
        except Exception as e:
            print(f"Error loading carry forward from MongoDB: {str(e)}")
            return None

    # --------------------------------------------------------------------------------------
    # History Operations (history_{account}.pkl)
    # --------------------------------------------------------------------------------------

    def save_history(self, account, df):
        """Save historical cleared/matched breaks for an account."""
        if not self.connected:
            return False
        try:
            collection = self.db['history']
            records = df.to_dict('records')
            doc = {
                'account': account,
                'data': records,
                'columns': list(df.columns),
                'updated_at': datetime.now()
            }
            collection.update_one(
                {'account': account},
                {'$set': doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving history to MongoDB: {str(e)}")
            return False

    def load_history(self, account):
        """Load historical cleared/matched breaks for an account. Returns DataFrame or None."""
        if not self.connected:
            return None
        try:
            collection = self.db['history']
            doc = collection.find_one({'account': account})
            if doc and 'data' in doc:
                return pd.DataFrame(doc['data'], columns=doc.get('columns', []))
            return None
        except Exception as e:
            print(f"Error loading history from MongoDB: {str(e)}")
            return None

    def append_history(self, account, new_df):
        """Append new rows to history for an account."""
        if not self.connected:
            return False
        try:
            existing_df = self.load_history(account)
            if existing_df is not None and not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            return self.save_history(account, combined_df)
        except Exception as e:
            print(f"Error appending history to MongoDB: {str(e)}")
            return False

    # --------------------------------------------------------------------------------------
    # Accounts List Operations (accounts.json)
    # --------------------------------------------------------------------------------------

    def save_accounts_list(self, accounts_list):
        """Save list of accounts."""
        if not self.connected:
            return False
        try:
            collection = self.db['accounts']
            doc = {
                '_id': 'accounts_list',
                'accounts': sorted(set([x for x in accounts_list if x])),
                'updated_at': datetime.now()
            }
            collection.update_one(
                {'_id': 'accounts_list'},
                {'$set': doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving accounts list to MongoDB: {str(e)}")
            return False

    def load_accounts_list(self):
        """Load list of accounts. Returns list or empty list."""
        if not self.connected:
            return []
        try:
            collection = self.db['accounts']
            doc = collection.find_one({'_id': 'accounts_list'})
            if doc and 'accounts' in doc:
                return doc['accounts']
            return []
        except Exception as e:
            print(f"Error loading accounts list from MongoDB: {str(e)}")
            return []

    # --------------------------------------------------------------------------------------
    # Accounts-by-Broker Mapping  { broker_key: [account, ...] }
    #
    # FIX: Previously this mapping was only stored in a local JSON file
    # (DATA_ROOT / "accounts_by_broker.json"), which is ephemeral in containers
    # and gets wiped on every Fly.io / Back4App / Render deployment.
    # Now it is persisted in MongoDB so it survives restarts.
    # --------------------------------------------------------------------------------------

    def save_accounts_by_broker(self, mapping: dict) -> bool:
        """
        Persist the { broker_key -> [account, ...] } mapping.

        Args:
            mapping: full dict to persist (replaces previous value)
        """
        if not self.connected:
            return False
        try:
            collection = self.db['accounts']
            doc = {
                '_id': 'accounts_by_broker',
                'mapping': mapping,
                'updated_at': datetime.now()
            }
            collection.update_one(
                {'_id': 'accounts_by_broker'},
                {'$set': doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving accounts_by_broker to MongoDB: {str(e)}")
            return False

    def load_accounts_by_broker(self) -> dict:
        """
        Load the { broker_key -> [account, ...] } mapping.

        Returns:
            dict or empty dict
        """
        if not self.connected:
            return {}
        try:
            collection = self.db['accounts']
            doc = collection.find_one({'_id': 'accounts_by_broker'})
            if doc and 'mapping' in doc:
                return doc['mapping']
            return {}
        except Exception as e:
            print(f"Error loading accounts_by_broker from MongoDB: {str(e)}")
            return {}

    # --------------------------------------------------------------------------------------
    # Session Metadata Operations
    # --------------------------------------------------------------------------------------

    def save_session_metadata(self, session_id, metadata):
        """Save session metadata."""
        if not self.connected:
            return False
        try:
            collection = self.db['session_metadata']
            doc = {
                'session_id': session_id,
                'metadata': metadata,
                'updated_at': datetime.now()
            }
            collection.update_one(
                {'session_id': session_id},
                {'$set': doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving session metadata to MongoDB: {str(e)}")
            return False

    def load_session_metadata(self, session_id):
        """Load session metadata. Returns dict or None."""
        if not self.connected:
            return None
        try:
            collection = self.db['session_metadata']
            doc = collection.find_one({'session_id': session_id})
            if doc and 'metadata' in doc:
                return doc['metadata']
            return None
        except Exception as e:
            print(f"Error loading session metadata from MongoDB: {str(e)}")
            return None

    # --------------------------------------------------------------------------------------
    # Utility Operations
    # --------------------------------------------------------------------------------------

    def delete_session_data(self, session_id):
        """Delete all data for a session."""
        if not self.connected:
            return False
        try:
            self.db['session_rec'].delete_one({'session_id': session_id})
            self.db['session_metadata'].delete_one({'session_id': session_id})
            return True
        except Exception as e:
            print(f"Error deleting session data from MongoDB: {str(e)}")
            return False

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.connected = False
            print("MongoDB connection closed")
