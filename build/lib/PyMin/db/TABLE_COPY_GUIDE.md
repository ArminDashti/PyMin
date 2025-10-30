# Table Copying Guide for PyMin MSSQL

This guide explains how to use the table copying functionality in the PyMin MSSQL class to copy tables between databases on the same server or different servers.

## Overview

The PyMin MSSQL class provides three main methods for copying tables:

1. `copy_table()` - Complete table copy (structure + data)
2. `copy_table_structure_only()` - Copy only table structure (schema)
3. `copy_table_data_only()` - Copy only data to existing table

## Features

- **Same Server**: Copy tables between different databases on the same SQL Server instance
- **Different Servers**: Copy tables between different SQL Server instances
- **Data Filtering**: Use WHERE clauses to filter data during copy
- **Structure Preservation**: Automatically recreates table structure with proper data types
- **Error Handling**: Comprehensive error handling and logging
- **Connection Management**: Automatic connection management for target databases

## Method Reference

### `copy_table()`

The main method for copying tables with full control over the process.

```python
def copy_table(self, source_table: str, target_table: str = None, 
               target_database: str = None, target_server: str = None,
               target_username: str = None, target_password: str = None,
               target_port: int = None, copy_data: bool = True,
               create_table: bool = True, where_clause: str = None) -> bool
```

**Parameters:**
- `source_table` (str): Name of the source table to copy
- `target_table` (str): Name of the target table (defaults to source_table)
- `target_database` (str): Target database name (defaults to current database)
- `target_server` (str): Target server IP/instance (defaults to current server)
- `target_username` (str): Target server username (defaults to current username)
- `target_password` (str): Target server password (defaults to current password)
- `target_port` (int): Target server port (defaults to current port)
- `copy_data` (bool): Whether to copy data (default: True)
- `create_table` (bool): Whether to create target table if it doesn't exist (default: True)
- `where_clause` (str): Optional WHERE clause to filter data during copy

**Returns:** `bool` - True if successful, False otherwise

### `copy_table_structure_only()`

Convenience method to copy only the table structure without data.

```python
def copy_table_structure_only(self, source_table: str, target_table: str = None,
                             target_database: str = None, target_server: str = None,
                             target_username: str = None, target_password: str = None,
                             target_port: int = None) -> bool
```

### `copy_table_data_only()`

Convenience method to copy only data to an existing table.

```python
def copy_table_data_only(self, source_table: str, target_table: str,
                        target_database: str = None, target_server: str = None,
                        target_username: str = None, target_password: str = None,
                        target_port: int = None, where_clause: str = None) -> bool
```

## Usage Examples

### 1. Copy Table Within Same Server

```python
from PyMin.db.mssql.mssql import MSSQL

# Source database connection
source_db = MSSQL(
    ip_or_instance="localhost",
    username="sa",
    password="YourPassword123!",
    database="SourceDatabase"
)

# Connect and copy table
if source_db.connect():
    success = source_db.copy_table(
        source_table="users",
        target_table="users_backup",
        target_database="BackupDatabase"
    )
    source_db.disconnect()
```

### 2. Copy Table to Different Server

```python
# Copy to different server
success = source_db.copy_table(
    source_table="products",
    target_table="products_mirror",
    target_database="TargetDatabase",
    target_server="target-server.com",
    target_username="admin",
    target_password="TargetPassword123!",
    target_port=1433
)
```

### 3. Copy Table Structure Only

```python
# Copy only table structure
success = source_db.copy_table_structure_only(
    source_table="orders",
    target_table="orders_template",
    target_database="TemplateDatabase"
)
```

### 4. Copy Table with Data Filtering

```python
# Copy with WHERE clause to filter data
success = source_db.copy_table(
    source_table="sales",
    target_table="sales_2024",
    target_database="ArchiveDatabase",
    where_clause="YEAR(sale_date) = 2024"
)
```

### 5. Copy Data Only to Existing Table

```python
# Copy only data to existing table
success = source_db.copy_table_data_only(
    source_table="inventory",
    target_table="inventory_backup",
    target_database="BackupDatabase",
    where_clause="quantity > 0"
)
```

### 6. Bulk Table Copy

```python
# Copy multiple tables
tables_to_copy = ["users", "products", "orders", "inventory"]

for table in tables_to_copy:
    if source_db.table_exists(table):
        success = source_db.copy_table(
            source_table=table,
            target_table=f"{table}_backup",
            target_database="BackupDatabase"
        )
        print(f"Table {table}: {'Success' if success else 'Failed'}")
```

## Data Type Support

The table copying functionality automatically handles various SQL Server data types:

- **String Types**: `varchar`, `nvarchar`, `char`, `nchar` (with proper length)
- **Numeric Types**: `int`, `bigint`, `decimal`, `numeric`, `float`, `real`
- **Date/Time Types**: `datetime`, `datetime2`, `date`, `time`
- **Binary Types**: `varbinary`, `binary`
- **Other Types**: `bit`, `uniqueidentifier`, `text`, `ntext`

## Error Handling

The methods include comprehensive error handling:

- **Connection Validation**: Checks source and target database connections
- **Table Existence**: Verifies source table exists before copying
- **Structure Validation**: Ensures table structure can be retrieved
- **Transaction Management**: Proper rollback on errors
- **Logging**: Detailed logging for debugging and monitoring

## Performance Considerations

- **Large Tables**: For very large tables, consider using data filtering with WHERE clauses
- **Network Latency**: Cross-server copies will be slower due to network overhead
- **Memory Usage**: Large datasets are loaded into memory; monitor memory usage
- **Transaction Size**: Consider breaking very large copies into smaller batches

## Security Considerations

- **Credentials**: Store database credentials securely (environment variables, key vaults)
- **Permissions**: Ensure source user has SELECT permissions and target user has CREATE/INSERT permissions
- **Network Security**: Use encrypted connections for cross-server copies
- **Data Sensitivity**: Be cautious when copying sensitive data between servers

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Verify server addresses and ports
   - Check firewall settings
   - Validate credentials

2. **Permission Errors**
   - Ensure proper database permissions
   - Check table-level permissions

3. **Data Type Issues**
   - Some complex data types may not be fully supported
   - Check logs for specific data type errors

4. **Memory Issues**
   - Use WHERE clauses to limit data size
   - Consider copying in batches for very large tables

### Logging

Enable detailed logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Test First**: Always test with small tables first
2. **Backup**: Create backups before copying critical data
3. **Validate**: Verify data integrity after copying
4. **Monitor**: Monitor performance and resource usage
5. **Document**: Document your copying procedures for future reference

## Example Script

See `table_copy_example.py` for complete working examples of all copying scenarios.
