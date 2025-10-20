using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Data.SqlClient;
using MySql.Data.MySqlClient;
using T1.EfCodeFirstGenerator.Models;

namespace T1.EfCodeFirstGenerator.SchemaExtractor
{
    internal class DatabaseSchemaExtractor
    {
        public static DbSchema CreateDatabaseSchema(DbConfig dbConfig)
        {
            var schema = new DbSchema
            {
                DatabaseName = dbConfig.DatabaseName
            };

            switch (dbConfig.DbType)
            {
                case DbType.SqlServer:
                    ExtractSqlServerSchema(dbConfig, schema);
                    break;
                case DbType.MySql:
                    ExtractMySqlSchema(dbConfig, schema);
                    break;
                default:
                    throw new NotSupportedException($"DbType {dbConfig.DbType} is not supported yet.");
            }

            return schema;
        }

        private static void ExtractSqlServerSchema(DbConfig dbConfig, DbSchema schema)
        {
            using (var connection = new SqlConnection(dbConfig.GetConnectionString()))
            {
                connection.Open();

                var tables = GetTables(connection, dbConfig.DatabaseName);
                
                foreach (var tableName in tables)
                {
                    var tableSchema = new TableSchema { TableName = tableName };
                    
                    var fields = GetSqlServerFields(connection, tableName);
                    tableSchema.Fields.AddRange(fields);
                    
                    schema.Tables.Add(tableSchema);
                }
            }
        }

        private static void ExtractMySqlSchema(DbConfig dbConfig, DbSchema schema)
        {
            using (var connection = new MySqlConnection(dbConfig.GetConnectionString()))
            {
                connection.Open();

                var tables = GetTables(connection, dbConfig.DatabaseName);
                
                foreach (var tableName in tables)
                {
                    var tableSchema = new TableSchema { TableName = tableName };
                    
                    var fields = GetMySqlFields(connection, dbConfig.DatabaseName, tableName);
                    tableSchema.Fields.AddRange(fields);
                    
                    schema.Tables.Add(tableSchema);
                }
            }
        }

        private static List<string> GetTables(DbConnection connection, string databaseName)
        {
            var tables = new List<string>();
            
            using (var command = connection.CreateCommand())
            {
                command.CommandText = @"
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE' 
                    AND (TABLE_CATALOG = @DatabaseName OR TABLE_SCHEMA = @DatabaseName)
                    ORDER BY TABLE_NAME";
                
                var param = command.CreateParameter();
                param.ParameterName = "@DatabaseName";
                param.Value = databaseName;
                command.Parameters.Add(param);

                using (var reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        tables.Add(reader.GetString(0));
                    }
                }
            }

            return tables;
        }

        private static List<FieldSchema> GetSqlServerFields(SqlConnection connection, string tableName)
        {
            var fields = new List<FieldSchema>();

            var query = @"
                SELECT 
                    c.COLUMN_NAME,
                    c.DATA_TYPE + 
                        CASE 
                            WHEN c.CHARACTER_MAXIMUM_LENGTH IS NOT NULL AND c.CHARACTER_MAXIMUM_LENGTH != -1 
                                THEN '(' + CAST(c.CHARACTER_MAXIMUM_LENGTH AS VARCHAR) + ')'
                            WHEN c.NUMERIC_PRECISION IS NOT NULL 
                                THEN '(' + CAST(c.NUMERIC_PRECISION AS VARCHAR) + ',' + CAST(c.NUMERIC_SCALE AS VARCHAR) + ')'
                            ELSE ''
                        END AS DATA_TYPE_FULL,
                    c.IS_NULLABLE,
                    c.COLUMN_DEFAULT,
                    CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END AS IS_PRIMARY_KEY
                FROM INFORMATION_SCHEMA.COLUMNS c
                LEFT JOIN (
                    SELECT ku.TABLE_NAME, ku.COLUMN_NAME
                    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc
                    INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS ku
                        ON tc.CONSTRAINT_TYPE = 'PRIMARY KEY' 
                        AND tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                ) pk ON c.TABLE_NAME = pk.TABLE_NAME AND c.COLUMN_NAME = pk.COLUMN_NAME
                WHERE c.TABLE_NAME = @TableName
                ORDER BY c.ORDINAL_POSITION";

            using (var command = new SqlCommand(query, connection))
            {
                command.Parameters.AddWithValue("@TableName", tableName);

                using (var reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        var field = new FieldSchema
                        {
                            FieldName = reader.GetString(0),
                            SqlDataType = reader.GetString(1),
                            IsNullable = reader.GetString(2).Equals("YES", StringComparison.OrdinalIgnoreCase),
                            DefaultValue = reader.IsDBNull(3) ? null : reader.GetString(3),
                            IsPrimaryKey = reader.GetInt32(4) == 1
                        };
                        fields.Add(field);
                    }
                }
            }

            return fields;
        }

        private static List<FieldSchema> GetMySqlFields(MySqlConnection connection, string databaseName, string tableName)
        {
            var fields = new List<FieldSchema>();

            var query = @"
                SELECT 
                    c.COLUMN_NAME,
                    c.COLUMN_TYPE,
                    c.IS_NULLABLE,
                    c.COLUMN_DEFAULT,
                    c.COLUMN_KEY
                FROM INFORMATION_SCHEMA.COLUMNS c
                WHERE c.TABLE_SCHEMA = @DatabaseName
                AND c.TABLE_NAME = @TableName
                ORDER BY c.ORDINAL_POSITION";

            using (var command = new MySqlCommand(query, connection))
            {
                command.Parameters.AddWithValue("@DatabaseName", databaseName);
                command.Parameters.AddWithValue("@TableName", tableName);

                using (var reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        var field = new FieldSchema
                        {
                            FieldName = reader.GetString(0),
                            SqlDataType = reader.GetString(1),
                            IsNullable = reader.GetString(2).Equals("YES", StringComparison.OrdinalIgnoreCase),
                            DefaultValue = reader.IsDBNull(3) ? null : reader.GetString(3),
                            IsPrimaryKey = reader.GetString(4).Equals("PRI", StringComparison.OrdinalIgnoreCase)
                        };
                        fields.Add(field);
                    }
                }
            }

            return fields;
        }
    }
}

