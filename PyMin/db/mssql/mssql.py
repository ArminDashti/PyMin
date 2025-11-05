"""
MSSQL Database Connection and Operations Class

This module provides a comprehensive interface for interacting with Microsoft SQL Server databases.
It includes basic CRUD operations, connection management, and additional utility methods.
"""

import pyodbc
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager
from datetime import datetime
import time


class MSSQL:
    """
    A comprehensive class for Microsoft SQL Server database operations.
    
    This class provides methods for connecting to MSSQL databases and performing
    various database operations including CRUD operations, data analysis, and
    database management tasks.
    """
    
    def __init__(self, ip_or_instance: str, username: str, password: str, 
                 database: str = "master", port: int = 1433, 
                 driver: str = "ODBC Driver 17 for SQL Server", log_path: str = None):
        """
        Initialize MSSQL connection parameters.
        
        Args:
            ip_or_instance (str): Server IP address or instance name
            username (str): Database username
            password (str): Database password
            database (str): Database name (default: "master")
            port (int): Port number (default: 1433)
            driver (str): ODBC driver name
            log_path (str): Path to log file for query logging (default: None)
        """
        self.ip_or_instance = ip_or_instance
        self.username = username
        self.password = password
        self.database = database
        self.port = port
        self.driver = driver
        self.connection = None
        self.log_path = log_path
    
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
    
    def _log_query(self, operation_name: str, query: str, execution_time: float, result=None, is_select: bool = False):
        if not self.log_path:
            return
        
        now = datetime.now()
        date_str = now.strftime("%y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"========= {operation_name} *** {self.ip_or_instance} *** {self.username} *** {self.database} *** {date_str} *** {time_str} =========\n")
            f.write(f"\n{query}\n\n")
            f.write(f"Execution time: {execution_time:.4f} seconds\n\n")
            
            if is_select and result is not None and not result.empty:
                first_row = result.iloc[0].to_dict()
                f.write(f"First row: {first_row}\n\n")
            elif not is_select and result is not None:
                f.write(f"Rows affected: {result}\n\n")
    
    def connect(self) -> bool:
        """
        Establish connection to the database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = pyodbc.connect(self._get_connection_string())
            return True
        except Exception as e:
            return False
    
    def disconnect(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
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
        
        start_time = time.time()
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query)
                columns = [column[0] for column in cursor.description]
                data = cursor.fetchall()
                result = pd.DataFrame(data, columns=columns)
                execution_time = time.time() - start_time
                self._log_query("SELECT", query, execution_time, result, is_select=True)
                return result
        except Exception as e:
            execution_time = time.time() - start_time
            self._log_query("SELECT", query, execution_time, None, is_select=True)
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
        
        start_time = time.time()
        try:
            with self.get_cursor() as cursor:
                rows_affected = 0
                for record in data:
                    values = [record.get(col) for col in columns]
                    cursor.execute(query, values)
                    rows_affected += cursor.rowcount
                
                self.connection.commit()
                execution_time = time.time() - start_time
                self._log_query("INSERT", query, execution_time, rows_affected, is_select=False)
                return rows_affected
        except Exception as e:
            self.connection.rollback()
            execution_time = time.time() - start_time
            self._log_query("INSERT", query, execution_time, None, is_select=False)
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
        
        start_time = time.time()
        try:
            with self.get_cursor() as cursor:
                values = list(data.values())
                cursor.execute(query, values)
                rows_affected = cursor.rowcount
                self.connection.commit()
                execution_time = time.time() - start_time
                self._log_query("UPDATE", query, execution_time, rows_affected, is_select=False)
                return rows_affected
        except Exception as e:
            self.connection.rollback()
            execution_time = time.time() - start_time
            self._log_query("UPDATE", query, execution_time, None, is_select=False)
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
        
        start_time = time.time()
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query)
                rows_affected = cursor.rowcount
                self.connection.commit()
                execution_time = time.time() - start_time
                self._log_query("DELETE", query, execution_time, rows_affected, is_select=False)
                return rows_affected
        except Exception as e:
            self.connection.rollback()
            execution_time = time.time() - start_time
            self._log_query("DELETE", query, execution_time, None, is_select=False)
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
        is_select = query.strip().upper().startswith('SELECT')
        start_time = time.time()
        try:
            with self.get_cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if is_select:
                    columns = [column[0] for column in cursor.description]
                    data = cursor.fetchall()
                    result = pd.DataFrame(data, columns=columns)
                    execution_time = time.time() - start_time
                    self._log_query("EXECUTE_RAW", query, execution_time, result, is_select=True)
                    return result
                else:
                    rows_affected = cursor.rowcount
                    self.connection.commit()
                    execution_time = time.time() - start_time
                    self._log_query("EXECUTE_RAW", query, execution_time, rows_affected, is_select=False)
                    return pd.DataFrame()
        except Exception as e:
            self.connection.rollback()
            execution_time = time.time() - start_time
            self._log_query("EXECUTE_RAW", query, execution_time, None, is_select=is_select)
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
    
    def get_all_tables(self, like: str = None) -> List[str]:
        """
        Get list of all tables in the database.
        
        Args:
            like (str): Optional pattern to filter table names (e.g., 'user%' for tables starting with 'user')
        
        Returns:
            List[str]: List of table names
        """
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        if like:
            query += " AND TABLE_NAME LIKE ?"
            result = self.execute_raw(query, (like,))
        else:
            result = self.execute_raw(query)
        return result['TABLE_NAME'].tolist()
    
    def get_all_databases(self) -> List[str]:
        """
        Get list of all databases on the server.
        
        Returns:
            List[str]: List of database names
        """
        query = "SELECT name FROM sys.databases WHERE database_id > 4"  # Exclude system databases
        result = self.execute_raw(query)
        return result['name'].tolist()
    
    def is_table_exist(self, table: str, like: str = None) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table (str): Table name
            like (str): Optional pattern to filter table names (e.g., 'user%' for tables starting with 'user')
        Returns:
            bool: True if table exists, False otherwise
        """
        query = "SELECT COUNT(*) as count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?"
        if like:
            query += " AND TABLE_NAME LIKE ?"
            result = self.execute_raw(query, (like,))
        else:
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

        if columns is None:
            raise ValueError("Columns dictionary must be provided.")

        column_definitions = [f"{col} {data_type}" for col, data_type in columns.items()]

        query = f"CREATE TABLE {table} ({', '.join(column_definitions)})"

        try:
            self.execute_raw(query)
            return True
        except Exception:
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
            return True
        except Exception as e:
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
            return True
        except Exception as e:
            return False
    
    def _copy_table_structure(self, source_table: str, target_table: str,
                               target_db, table_info: pd.DataFrame) -> bool:
        target_exists = target_db.is_table_exist(target_table)
        
        if not target_exists:
            column_definitions = []
            for _, row in table_info.iterrows():
                col_name = row['COLUMN_NAME']
                data_type = row['DATA_TYPE']
                is_nullable = row['IS_NULLABLE']
                char_length = row['CHARACTER_MAXIMUM_LENGTH']
                numeric_precision = row['NUMERIC_PRECISION']
                numeric_scale = row['NUMERIC_SCALE']
                
                if data_type in ['varchar', 'nvarchar', 'char', 'nchar'] and char_length:
                    if char_length == -1:
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
            target_db.execute_raw(create_query)
        
        return True
    
    def copy_table_with_data(self, source_table: str, target_table: str = None, 
                             target_database: str = None, target_server: str = None,
                             target_username: str = None, target_password: str = None,
                             target_port: int = 1433, where_clause: str = None) -> bool:
        """
        Copy a table from source to target database with data.
        
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
            where_clause (str): Optional WHERE clause to filter data during copy
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
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
            
            if not self.is_table_exist(source_table):
                return False
            
            table_info = self.get_table_info(source_table)
            if table_info.empty:
                return False
            
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
                    return False
            else:
                target_db = self
            
            try:
                if not self._copy_table_structure(source_table, target_table, target_db, table_info):
                    return False
                
                columns = table_info['COLUMN_NAME'].tolist()
                select_query = f"SELECT {', '.join(columns)} FROM {source_table}"
                
                if where_clause:
                    select_query += f" WHERE {where_clause}"
                
                source_data = self.execute_raw(select_query)
                
                if not source_data.empty:
                    data_records = source_data.to_dict('records')
                    target_db.insert(target_table, data_records)
                
                return True
                
            finally:
                if target_db != self:
                    target_db.disconnect()
                    
        except Exception as e:
            return False
    
    def copy_table_without_data(self, source_table: str, target_table: str = None, 
                                target_database: str = None, target_server: str = None,
                                target_username: str = None, target_password: str = None,
                                target_port: int = 1433) -> bool:
        """
        Copy a table structure from source to target database without data.
        
        This method can copy table structures within the same database, between different databases
        on the same server, or between different servers entirely.
        
        Args:
            source_table (str): Name of the source table to copy
            target_table (str): Name of the target table (defaults to source_table)
            target_database (str): Target database name (defaults to current database)
            target_server (str): Target server IP/instance (defaults to current server)
            target_username (str): Target server username (defaults to current username)
            target_password (str): Target server password (defaults to current password)
            target_port (int): Target server port (defaults to current port)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
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
            
            if not self.is_table_exist(source_table):
                return False
            
            table_info = self.get_table_info(source_table)
            if table_info.empty:
                return False
            
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
                    return False
            else:
                target_db = self
            
            try:
                return self._copy_table_structure(source_table, target_table, target_db, table_info)
                
            finally:
                if target_db != self:
                    target_db.disconnect()
                    
        except Exception as e:
            return False
    
    def df_to_table(self, df: pd.DataFrame, table: str, if_exists: str = "fail") -> bool:
        if df.empty:
            return False
        
        if if_exists == "fail" and self.is_table_exist(table):
            return False
        
        if if_exists == "replace":
            self.drop_table(table)
        
        dtype_map = {
            'int64': 'BIGINT',
            'int32': 'INT',
            'int16': 'SMALLINT',
            'int8': 'TINYINT',
            'float64': 'FLOAT',
            'float32': 'REAL',
            'bool': 'BIT',
            'datetime64[ns]': 'DATETIME2',
            'object': 'NVARCHAR(MAX)'
        }
        
        columns = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            sql_type = dtype_map.get(dtype, 'NVARCHAR(MAX)')
            if dtype.startswith('datetime'):
                sql_type = 'DATETIME2'
            columns[col] = sql_type
        
        if not self.is_table_exist(table):
            self.create_table(table, columns)
        
        data_records = df.to_dict('records')
        self.insert(table, data_records)
        return True
    
    def export(self, tables_names: Union[str, List[str]], path: str, export_format: str = "csv") -> bool:
        if isinstance(tables_names, str):
            tables_names = [tables_names]
        
        if export_format not in ["csv", "df"]:
            return False
        
        for table_name in tables_names:
            df = self.select(table_name)
            
            if export_format == "csv":
                file_path = f"{path}/{table_name}.csv" if not path.endswith('.csv') else path
                df.to_csv(file_path, index=False)
            elif export_format == "df":
                file_path = f"{path}/{table_name}.pkl" if not path.endswith('.pkl') else path
                df.to_pickle(file_path)
        
        return True
    
