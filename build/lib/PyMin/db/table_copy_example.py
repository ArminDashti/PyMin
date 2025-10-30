"""
Example usage of table copying functionality in PyMin MSSQL class.

This example demonstrates how to copy tables between databases on the same server
or different servers using the MSSQL class.
"""

from PyMin.db.mssql.mssql import MSSQL
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def example_copy_table_same_server():
    """Example: Copy table within the same server but different database."""
    print("=== Example 1: Copy table within same server, different database ===")
    
    # Source database connection
    source_db = MSSQL(
        ip_or_instance="localhost",
        username="sa",
        password="YourPassword123!",
        database="SourceDatabase"
    )
    
    try:
        # Connect to source database
        if not source_db.connect():
            print("Failed to connect to source database")
            return
        
        # Copy table to different database on same server
        success = source_db.copy_table(
            source_table="users",
            target_table="users_backup",
            target_database="TargetDatabase"
        )
        
        if success:
            print("Table copied successfully!")
        else:
            print("Failed to copy table")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        source_db.disconnect()

def example_copy_table_different_server():
    """Example: Copy table to different server."""
    print("\n=== Example 2: Copy table to different server ===")
    
    # Source database connection
    source_db = MSSQL(
        ip_or_instance="source-server.com",
        username="sa",
        password="SourcePassword123!",
        database="SourceDatabase"
    )
    
    try:
        # Connect to source database
        if not source_db.connect():
            print("Failed to connect to source database")
            return
        
        # Copy table to different server
        success = source_db.copy_table(
            source_table="products",
            target_table="products_mirror",
            target_database="TargetDatabase",
            target_server="target-server.com",
            target_username="admin",
            target_password="TargetPassword123!",
            target_port=1433
        )
        
        if success:
            print("Table copied to different server successfully!")
        else:
            print("Failed to copy table to different server")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        source_db.disconnect()

def example_copy_table_structure_only():
    """Example: Copy only table structure without data."""
    print("\n=== Example 3: Copy table structure only ===")
    
    source_db = MSSQL(
        ip_or_instance="localhost",
        username="sa",
        password="YourPassword123!",
        database="SourceDatabase"
    )
    
    try:
        if not source_db.connect():
            print("Failed to connect to source database")
            return
        
        # Copy only table structure
        success = source_db.copy_table_structure_only(
            source_table="orders",
            target_table="orders_template",
            target_database="TemplateDatabase"
        )
        
        if success:
            print("Table structure copied successfully!")
        else:
            print("Failed to copy table structure")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        source_db.disconnect()

def example_copy_table_with_filter():
    """Example: Copy table with data filtering."""
    print("\n=== Example 4: Copy table with data filtering ===")
    
    source_db = MSSQL(
        ip_or_instance="localhost",
        username="sa",
        password="YourPassword123!",
        database="SourceDatabase"
    )
    
    try:
        if not source_db.connect():
            print("Failed to connect to source database")
            return
        
        # Copy table with WHERE clause to filter data
        success = source_db.copy_table(
            source_table="sales",
            target_table="sales_2024",
            target_database="ArchiveDatabase",
            where_clause="YEAR(sale_date) = 2024"
        )
        
        if success:
            print("Filtered table data copied successfully!")
        else:
            print("Failed to copy filtered table data")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        source_db.disconnect()

def example_copy_table_data_only():
    """Example: Copy only data to existing table."""
    print("\n=== Example 5: Copy data only to existing table ===")
    
    source_db = MSSQL(
        ip_or_instance="localhost",
        username="sa",
        password="YourPassword123!",
        database="SourceDatabase"
    )
    
    try:
        if not source_db.connect():
            print("Failed to connect to source database")
            return
        
        # Copy only data to existing table
        success = source_db.copy_table_data_only(
            source_table="inventory",
            target_table="inventory_backup",
            target_database="BackupDatabase",
            where_clause="quantity > 0"
        )
        
        if success:
            print("Data copied to existing table successfully!")
        else:
            print("Failed to copy data to existing table")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        source_db.disconnect()

def example_bulk_table_copy():
    """Example: Copy multiple tables at once."""
    print("\n=== Example 6: Bulk table copy ===")
    
    source_db = MSSQL(
        ip_or_instance="localhost",
        username="sa",
        password="YourPassword123!",
        database="SourceDatabase"
    )
    
    try:
        if not source_db.connect():
            print("Failed to connect to source database")
            return
        
        # List of tables to copy
        tables_to_copy = ["users", "products", "orders", "inventory"]
        
        print(f"Copying {len(tables_to_copy)} tables...")
        
        for table in tables_to_copy:
            if source_db.table_exists(table):
                success = source_db.copy_table(
                    source_table=table,
                    target_table=f"{table}_backup",
                    target_database="BackupDatabase"
                )
                
                if success:
                    print(f"✓ {table} copied successfully")
                else:
                    print(f"✗ Failed to copy {table}")
            else:
                print(f"⚠ Table {table} does not exist, skipping...")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        source_db.disconnect()

if __name__ == "__main__":
    print("PyMin MSSQL Table Copying Examples")
    print("=" * 50)
    
    # Note: Update connection parameters before running
    print("Note: Update connection parameters in the examples before running!")
    print("Make sure you have the necessary permissions and that target databases exist.")
    print()
    
    # Uncomment the examples you want to run:
    
    # example_copy_table_same_server()
    # example_copy_table_different_server()
    # example_copy_table_structure_only()
    # example_copy_table_with_filter()
    # example_copy_table_data_only()
    # example_bulk_table_copy()
    
    print("\nAll examples completed!")
