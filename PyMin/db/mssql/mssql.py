"""
MSSQL Database Connection and Operations Class

This module provides a comprehensive interface for interacting with Microsoft SQL Server databases.
It includes basic CRUD operations, connection management, and additional utility methods.
"""

import pyodbc
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging
from contextlib import contextmanager


class MSSQL:
    """
    A comprehensive class for Microsoft SQL Server database operations.
    
    This class provides methods for connecting to MSSQL databases and performing
    various database operations including CRUD operations, data analysis, and
    database management tasks.
    """
    
    def __init__(self, ip_or_instance: str, username: str, password: str, 
                 database: str = "master", port: int = 1433, 
                 driver: str = "ODBC Driver 17 for SQL Server"):
        """
        Initialize MSSQL connection parameters.
        
        Args:
            ip_or_instance (str): Server IP address or instance name
            username (str): Database username
            password (str): Database password
            database (str): Database name (default: "master")
            port (int): Port number (default: 1433)
            driver (str): ODBC driver name
        """
        self.ip_or_instance = ip_or_instance
        self.username = username
        self.password = password
        self.database = database
        self.port = port
        self.driver = driver
        self.connection = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_connection_string(self) -> str:
        """Generate connection string for MSSQL database."""
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.ip_or_instance},{self.port};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            "TrustServerCertificate=yes;"
        )
    
    def connect(self) -> bool:
        """
        Establish connection to the database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = pyodbc.connect(self._get_connection_string())
            self.logger.info(f"Successfully connected to {self.database} on {self.ip_or_instance}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            return False
    
    def disconnect(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Database connection closed")
    
    def get_connection(self):
        """
        Return the current database connection.
        
        Returns:
            pyodbc.Connection: Database connection object
        """
        if not self.connection:
            self.connect()
        return self.connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor."""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        try:
            yield cursor
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"Database operation failed: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def select(self, table: str, columns: str = "*", where: str = None, 
               order_by: str = None, group_by: str = None, limit: int = None) -> pd.DataFrame:
        """
        Execute SELECT query with optional WHERE, ORDER BY, and GROUP BY clauses.
        
        Args:
            table (str): Table name
            columns (str): Columns to select (default: "*")
            where (str): WHERE clause (without WHERE keyword)
            order_by (str): ORDER BY clause (without ORDER BY keyword)
            group_by (str): GROUP BY clause (without GROUP BY keyword)
            limit (int): Maximum number of rows to return
            
        Returns:
            pd.DataFrame: Query results as pandas DataFrame
        """
        query = f"SELECT {columns} FROM {table}"
        
        if where:
            query += f" WHERE {where}"
        
        if group_by:
            query += f" GROUP BY {group_by}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query)
                columns = [column[0] for column in cursor.description]
                data = cursor.fetchall()
                return pd.DataFrame(data, columns=columns)
        except Exception as e:
            self.logger.error(f"SELECT query failed: {str(e)}")
            raise
    
    def insert(self, table: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> int:
        """
        Insert data into a table.
        
        Args:
            table (str): Table name
            data (Union[Dict, List[Dict]]): Data to insert (single record or list of records)
            
        Returns:
            int: Number of rows affected
        """
        if isinstance(data, dict):
            data = [data]
        
        if not data:
            return 0
        
        columns = list(data[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        try:
            with self.get_cursor() as cursor:
                rows_affected = 0
                for record in data:
                    values = [record.get(col) for col in columns]
                    cursor.execute(query, values)
                    rows_affected += cursor.rowcount
                
                self.connection.commit()
                self.logger.info(f"Inserted {rows_affected} rows into {table}")
                return rows_affected
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"INSERT query failed: {str(e)}")
            raise
    
    def update(self, table: str, data: Dict[str, Any], where: str) -> int:
        """
        Update records in a table.
        
        Args:
            table (str): Table name
            data (Dict[str, Any]): Data to update
            where (str): WHERE clause (without WHERE keyword)
            
        Returns:
            int: Number of rows affected
        """
        set_clause = ", ".join([f"{col} = ?" for col in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        
        try:
            with self.get_cursor() as cursor:
                values = list(data.values())
                cursor.execute(query, values)
                rows_affected = cursor.rowcount
                self.connection.commit()
                self.logger.info(f"Updated {rows_affected} rows in {table}")
                return rows_affected
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"UPDATE query failed: {str(e)}")
            raise
    
    def delete(self, table: str, where: str) -> int:
        """
        Delete records from a table.
        
        Args:
            table (str): Table name
            where (str): WHERE clause (without WHERE keyword)
            
        Returns:
            int: Number of rows affected
        """
        query = f"DELETE FROM {table} WHERE {where}"
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query)
                rows_affected = cursor.rowcount
                self.connection.commit()
                self.logger.info(f"Deleted {rows_affected} rows from {table}")
                return rows_affected
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"DELETE query failed: {str(e)}")
            raise
    
    def execute_raw(self, query: str, params: tuple = None) -> pd.DataFrame:
        """
        Execute raw SQL query.
        
        Args:
            query (str): SQL query to execute
            params (tuple): Query parameters
            
        Returns:
            pd.DataFrame: Query results as pandas DataFrame
        """
        try:
            with self.get_cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    columns = [column[0] for column in cursor.description]
                    data = cursor.fetchall()
                    return pd.DataFrame(data, columns=columns)
                else:
                    self.connection.commit()
                    return pd.DataFrame()
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"Raw query execution failed: {str(e)}")
            raise
    
    def get_table_info(self, table: str) -> pd.DataFrame:
        """
        Get information about table structure.
        
        Args:
            table (str): Table name
            
        Returns:
            pd.DataFrame: Table structure information
        """
        query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
        """
        return self.execute_raw(query, (table,))
    
    def get_tables(self) -> List[str]:
        """
        Get list of all tables in the database.
        
        Returns:
            List[str]: List of table names
        """
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        result = self.execute_raw(query)
        return result['TABLE_NAME'].tolist()
    
    def get_databases(self) -> List[str]:
        """
        Get list of all databases on the server.
        
        Returns:
            List[str]: List of database names
        """
        query = "SELECT name FROM sys.databases WHERE database_id > 4"  # Exclude system databases
        result = self.execute_raw(query)
        return result['name'].tolist()
    
    def table_exists(self, table: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table (str): Table name
            
        Returns:
            bool: True if table exists, False otherwise
        """
        query = "SELECT COUNT(*) as count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?"
        result = self.execute_raw(query, (table,))
        return result['count'].iloc[0] > 0
    
    def get_table_row_count(self, table: str) -> int:
        """
        Get the number of rows in a table.
        
        Args:
            table (str): Table name
            
        Returns:
            int: Number of rows
        """
        query = f"SELECT COUNT(*) as count FROM {table}"
        result = self.execute_raw(query)
        return result['count'].iloc[0]
    
    def create_table(self, table: str, columns: Dict[str, str]) -> bool:
        """
        Create a new table.
        
        Args:
            table (str): Table name
            columns (Dict[str, str]): Column definitions (column_name: data_type)
            
        Returns:
            bool: True if successful, False otherwise
        """
        column_definitions = [f"{col} {data_type}" for col, data_type in columns.items()]
        query = f"CREATE TABLE {table} ({', '.join(column_definitions)})"
        
        try:
            self.execute_raw(query)
            self.logger.info(f"Table {table} created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create table {table}: {str(e)}")
            return False
    
    def drop_table(self, table: str) -> bool:
        """
        Drop a table.
        
        Args:
            table (str): Table name
            
        Returns:
            bool: True if successful, False otherwise
        """
        query = f"DROP TABLE {table}"
        
        try:
            self.execute_raw(query)
            self.logger.info(f"Table {table} dropped successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to drop table {table}: {str(e)}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create a database backup.
        
        Args:
            backup_path (str): Path where backup should be saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        query = f"BACKUP DATABASE {self.database} TO DISK = '{backup_path}'"
        
        try:
            self.execute_raw(query)
            self.logger.info(f"Database {self.database} backed up to {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup database: {str(e)}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def copy_table(self, source_table: str, target_table: str = None, 
                   target_database: str = None, target_server: str = None,
                   target_username: str = None, target_password: str = None,
                   target_port: int = None, copy_data: bool = True,
                   create_table: bool = True, where_clause: str = None) -> bool:
        """
        Copy a table from source to target database.
        
        This method can copy tables within the same database, between different databases
        on the same server, or between different servers entirely.
        
        Args:
            source_table (str): Name of the source table to copy
            target_table (str): Name of the target table (defaults to source_table)
            target_database (str): Target database name (defaults to current database)
            target_server (str): Target server IP/instance (defaults to current server)
            target_username (str): Target server username (defaults to current username)
            target_password (str): Target server password (defaults to current password)
            target_port (int): Target server port (defaults to current port)
            copy_data (bool): Whether to copy data (default: True)
            create_table (bool): Whether to create target table if it doesn't exist (default: True)
            where_clause (str): Optional WHERE clause to filter data during copy
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set defaults
            if target_table is None:
                target_table = source_table
            if target_database is None:
                target_database = self.database
            if target_server is None:
                target_server = self.ip_or_instance
            if target_username is None:
                target_username = self.username
            if target_password is None:
                target_password = self.password
            if target_port is None:
                target_port = self.port
            
            # Check if source table exists
            if not self.table_exists(source_table):
                self.logger.error(f"Source table '{source_table}' does not exist")
                return False
            
            # Get source table structure
            table_info = self.get_table_info(source_table)
            if table_info.empty:
                self.logger.error(f"Could not retrieve structure for table '{source_table}'")
                return False
            
            # Create target connection if different server/database
            target_db = None
            if (target_server != self.ip_or_instance or 
                target_database != self.database or 
                target_username != self.username or 
                target_password != self.password or 
                target_port != self.port):
                
                target_db = MSSQL(
                    ip_or_instance=target_server,
                    username=target_username,
                    password=target_password,
                    database=target_database,
                    port=target_port,
                    driver=self.driver
                )
                
                if not target_db.connect():
                    self.logger.error(f"Failed to connect to target database '{target_database}' on '{target_server}'")
                    return False
            else:
                target_db = self
            
            try:
                # Check if target table exists
                target_exists = target_db.table_exists(target_table)
                
                if create_table and not target_exists:
                    # Create target table with same structure
                    column_definitions = []
                    for _, row in table_info.iterrows():
                        col_name = row['COLUMN_NAME']
                        data_type = row['DATA_TYPE']
                        is_nullable = row['IS_NULLABLE']
                        char_length = row['CHARACTER_MAXIMUM_LENGTH']
                        numeric_precision = row['NUMERIC_PRECISION']
                        numeric_scale = row['NUMERIC_SCALE']
                        
                        # Build column definition
                        if data_type in ['varchar', 'nvarchar', 'char', 'nchar'] and char_length:
                            if char_length == -1:  # MAX
                                col_def = f"{col_name} {data_type}(MAX)"
                            else:
                                col_def = f"{col_name} {data_type}({char_length})"
                        elif data_type in ['decimal', 'numeric'] and numeric_precision and numeric_scale:
                            col_def = f"{col_name} {data_type}({numeric_precision},{numeric_scale})"
                        else:
                            col_def = f"{col_name} {data_type}"
                        
                        if is_nullable == 'NO':
                            col_def += " NOT NULL"
                        
                        column_definitions.append(col_def)
                    
                    create_query = f"CREATE TABLE {target_table} ({', '.join(column_definitions)})"
                    
                    if target_db.execute_raw(create_query).empty is False:
                        self.logger.info(f"Created target table '{target_table}' in database '{target_database}'")
                    else:
                        self.logger.info(f"Target table '{target_table}' created successfully")
                
                # Copy data if requested
                if copy_data:
                    # Build SELECT query for source data
                    columns = table_info['COLUMN_NAME'].tolist()
                    select_query = f"SELECT {', '.join(columns)} FROM {source_table}"
                    
                    if where_clause:
                        select_query += f" WHERE {where_clause}"
                    
                    # Get source data
                    source_data = self.execute_raw(select_query)
                    
                    if not source_data.empty:
                        # Convert DataFrame to list of dictionaries for insert
                        data_records = source_data.to_dict('records')
                        
                        # Insert data into target table
                        rows_inserted = target_db.insert(target_table, data_records)
                        self.logger.info(f"Copied {rows_inserted} rows from '{source_table}' to '{target_table}'")
                    else:
                        self.logger.info(f"No data to copy from '{source_table}' (empty result set)")
                
                self.logger.info(f"Successfully copied table '{source_table}' to '{target_table}'")
                return True
                
            finally:
                # Close target connection if it was created
                if target_db != self:
                    target_db.disconnect()
                    
        except Exception as e:
            self.logger.error(f"Failed to copy table '{source_table}': {str(e)}")
            return False
    
    def copy_table_structure_only(self, source_table: str, target_table: str = None,
                                 target_database: str = None, target_server: str = None,
                                 target_username: str = None, target_password: str = None,
                                 target_port: int = None) -> bool:
        """
        Copy only the table structure (schema) without data.
        
        Args:
            source_table (str): Name of the source table
            target_table (str): Name of the target table (defaults to source_table)
            target_database (str): Target database name (defaults to current database)
            target_server (str): Target server IP/instance (defaults to current server)
            target_username (str): Target server username (defaults to current username)
            target_password (str): Target server password (defaults to current password)
            target_port (int): Target server port (defaults to current port)
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.copy_table(
            source_table=source_table,
            target_table=target_table,
            target_database=target_database,
            target_server=target_server,
            target_username=target_username,
            target_password=target_password,
            target_port=target_port,
            copy_data=False,
            create_table=True
        )
    
    def copy_table_data_only(self, source_table: str, target_table: str,
                            target_database: str = None, target_server: str = None,
                            target_username: str = None, target_password: str = None,
                            target_port: int = None, where_clause: str = None) -> bool:
        """
        Copy only the data from source table to existing target table.
        
        Args:
            source_table (str): Name of the source table
            target_table (str): Name of the target table (must exist)
            target_database (str): Target database name (defaults to current database)
            target_server (str): Target server IP/instance (defaults to current server)
            target_username (str): Target server username (defaults to current username)
            target_password (str): Target server password (defaults to current password)
            target_port (int): Target server port (defaults to current port)
            where_clause (str): Optional WHERE clause to filter data during copy
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.copy_table(
            source_table=source_table,
            target_table=target_table,
            target_database=target_database,
            target_server=target_server,
            target_username=target_username,
            target_password=target_password,
            target_port=target_port,
            copy_data=True,
            create_table=False,
            where_clause=where_clause
        )

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.disconnect()
