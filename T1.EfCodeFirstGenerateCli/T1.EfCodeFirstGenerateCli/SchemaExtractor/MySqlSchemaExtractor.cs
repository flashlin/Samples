using System;
using System.Collections.Generic;
using MySql.Data.MySqlClient;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCli.SchemaExtractor
{
    internal class MySqlSchemaExtractor : ISchemaExtractor
    {
        public DbSchema ExtractSchema(DbConfig dbConfig)
        {
            var schema = new DbSchema
            {
                DatabaseName = dbConfig.DatabaseName
            };

            using (var connection = new MySqlConnection(dbConfig.GetConnectionString()))
            {
                connection.Open();

                var tables = SchemaExtractorHelper.GetTables(connection, dbConfig.DatabaseName);
                
                foreach (var tableName in tables)
                {
                    var tableSchema = new TableSchema { TableName = tableName };
                    
                    var fields = GetFields(connection, dbConfig.DatabaseName, tableName);
                    tableSchema.Fields.AddRange(fields);
                    
                    schema.Tables.Add(tableSchema);
                }
            }

            return schema;
        }

        private List<FieldSchema> GetFields(MySqlConnection connection, string databaseName, string tableName)
        {
            var fields = new List<FieldSchema>();

            var query = @"
                SELECT 
                    c.COLUMN_NAME,
                    c.COLUMN_TYPE,
                    c.IS_NULLABLE,
                    c.COLUMN_DEFAULT,
                    c.COLUMN_KEY,
                    c.EXTRA
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
                        var extraInfo = reader.IsDBNull(5) ? string.Empty : reader.GetString(5);
                        var field = new FieldSchema
                        {
                            FieldName = reader.IsDBNull(0) ? string.Empty : reader.GetString(0),
                            SqlDataType = reader.IsDBNull(1) ? string.Empty : reader.GetString(1),
                            IsNullable = reader.IsDBNull(2) ? false : reader.GetString(2).Equals("YES", StringComparison.OrdinalIgnoreCase),
                            DefaultValue = reader.IsDBNull(3) ? null : reader.GetString(3),
                            IsPrimaryKey = reader.IsDBNull(4) ? false : reader.GetString(4).Equals("PRI", StringComparison.OrdinalIgnoreCase),
                            IsAutoIncrement = extraInfo.IndexOf("auto_increment", StringComparison.OrdinalIgnoreCase) >= 0
                        };
                        fields.Add(field);
                    }
                }
            }

            return fields;
        }
    }
}

