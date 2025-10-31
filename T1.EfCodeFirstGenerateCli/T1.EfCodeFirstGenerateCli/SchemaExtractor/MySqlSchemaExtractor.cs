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
                DatabaseName = dbConfig.DatabaseName,
                ContextName = dbConfig.ContextName
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
                    c.EXTRA,
                    c.GENERATION_EXPRESSION
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
                        var generationExpression = reader.IsDBNull(6) ? null : reader.GetString(6);
                        
                        var field = new FieldSchema
                        {
                            FieldName = reader.IsDBNull(0) ? string.Empty : reader.GetString(0),
                            SqlDataType = reader.IsDBNull(1) ? string.Empty : reader.GetString(1),
                            IsNullable = reader.IsDBNull(2) ? false : reader.GetString(2).Equals("YES", StringComparison.OrdinalIgnoreCase),
                            DefaultValue = reader.IsDBNull(3) ? null : reader.GetString(3),
                            IsPrimaryKey = reader.IsDBNull(4) ? false : reader.GetString(4).Equals("PRI", StringComparison.OrdinalIgnoreCase),
                            IsAutoIncrement = extraInfo.IndexOf("auto_increment", StringComparison.OrdinalIgnoreCase) >= 0,
                            IsComputed = !string.IsNullOrEmpty(generationExpression),
                            ComputedColumnSql = generationExpression,
                            IsComputedColumnStored = extraInfo.IndexOf("STORED GENERATED", StringComparison.OrdinalIgnoreCase) >= 0
                        };
                        fields.Add(field);
                    }
                }
            }

            return fields;
        }

        public List<EntityRelationship> ExtractRelationships(DbConfig dbConfig)
        {
            var relationships = new List<EntityRelationship>();

            using (var connection = new MySqlConnection(dbConfig.GetConnectionString()))
            {
                connection.Open();

                // Step 1: Collect all foreign key data first
                var foreignKeyData = new List<(string DependentTable, string ForeignKeyColumn, string PrincipalTable, string PrincipalKeyColumn, bool IsNullable)>();

                var query = @"
                    SELECT 
                        kcu.TABLE_NAME AS DependentTable,
                        kcu.COLUMN_NAME AS ForeignKeyColumn,
                        kcu.REFERENCED_TABLE_NAME AS PrincipalTable,
                        kcu.REFERENCED_COLUMN_NAME AS PrincipalKeyColumn,
                        c.IS_NULLABLE AS IsNullable
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                    JOIN INFORMATION_SCHEMA.COLUMNS c 
                        ON kcu.TABLE_SCHEMA = c.TABLE_SCHEMA 
                        AND kcu.TABLE_NAME = c.TABLE_NAME 
                        AND kcu.COLUMN_NAME = c.COLUMN_NAME
                    WHERE kcu.CONSTRAINT_SCHEMA = @DatabaseName 
                        AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
                    ORDER BY DependentTable, ForeignKeyColumn";

                using (var command = new MySqlCommand(query, connection))
                {
                    command.Parameters.AddWithValue("@DatabaseName", dbConfig.DatabaseName);

                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            foreignKeyData.Add((
                                reader.GetString(0),
                                reader.GetString(1),
                                reader.GetString(2),
                                reader.GetString(3),
                                reader.GetString(4).Equals("YES", StringComparison.OrdinalIgnoreCase)
                            ));
                        }
                    }
                }

                // Step 2: Process collected data and determine relationship types
                foreach (var fk in foreignKeyData)
                {
                    var relType = DetermineRelationshipType(connection, dbConfig.DatabaseName, fk.DependentTable, fk.ForeignKeyColumn);

                    relationships.Add(new EntityRelationship
                    {
                        PrincipalEntity = fk.PrincipalTable,
                        PrincipalKey = fk.PrincipalKeyColumn,
                        DependentEntity = fk.DependentTable,
                        ForeignKey = fk.ForeignKeyColumn,
                        Type = relType,
                        NavigationType = NavigationType.Bidirectional,
                        IsPrincipalOptional = false,
                        IsDependentOptional = fk.IsNullable
                    });
                }
            }

            return relationships;
        }

        private RelationshipType DetermineRelationshipType(MySqlConnection connection, string databaseName, string tableName, string columnName)
        {
            // Check if the foreign key column is part of a unique constraint or primary key
            var query = @"
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
                    ON tc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA 
                    AND tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                    AND tc.TABLE_NAME = kcu.TABLE_NAME
                WHERE tc.CONSTRAINT_TYPE IN ('PRIMARY KEY', 'UNIQUE')
                    AND tc.TABLE_SCHEMA = @DatabaseName
                    AND tc.TABLE_NAME = @TableName
                    AND kcu.COLUMN_NAME = @ColumnName";

            using (var command = new MySqlCommand(query, connection))
            {
                command.Parameters.AddWithValue("@DatabaseName", databaseName);
                command.Parameters.AddWithValue("@TableName", tableName);
                command.Parameters.AddWithValue("@ColumnName", columnName);

                var count = Convert.ToInt32(command.ExecuteScalar());
                return count > 0 ? RelationshipType.OneToOne : RelationshipType.OneToMany;
            }
        }
    }
}

