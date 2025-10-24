using System;
using System.Collections.Generic;
using Microsoft.Data.SqlClient;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCli.SchemaExtractor
{
    internal class SqlServerSchemaExtractor : ISchemaExtractor
    {
        public DbSchema ExtractSchema(DbConfig dbConfig)
        {
            var schema = new DbSchema
            {
                DatabaseName = dbConfig.DatabaseName,
                ContextName = dbConfig.ContextName
            };

            using (var connection = new SqlConnection(dbConfig.GetConnectionString()))
            {
                connection.Open();

                var tables = SchemaExtractorHelper.GetTables(connection, dbConfig.DatabaseName);
                
                foreach (var tableName in tables)
                {
                    var tableSchema = new TableSchema { TableName = tableName };
                    
                    var fields = GetFields(connection, tableName);
                    tableSchema.Fields.AddRange(fields);
                    
                    schema.Tables.Add(tableSchema);
                }
            }

            return schema;
        }

        private List<FieldSchema> GetFields(SqlConnection connection, string tableName)
        {
            var fields = new List<FieldSchema>();

            var query = @"
                SELECT 
                    c.COLUMN_NAME,
                    c.DATA_TYPE + 
                        CASE 
                            WHEN c.CHARACTER_MAXIMUM_LENGTH IS NOT NULL AND c.CHARACTER_MAXIMUM_LENGTH != -1 
                                THEN '(' + CAST(c.CHARACTER_MAXIMUM_LENGTH AS VARCHAR) + ')'
                            WHEN c.NUMERIC_PRECISION IS NOT NULL AND c.NUMERIC_SCALE IS NOT NULL
                                THEN '(' + CAST(c.NUMERIC_PRECISION AS VARCHAR) + ',' + CAST(c.NUMERIC_SCALE AS VARCHAR) + ')'
                            WHEN c.NUMERIC_PRECISION IS NOT NULL AND c.NUMERIC_SCALE IS NULL
                                THEN '(' + CAST(c.NUMERIC_PRECISION AS VARCHAR) + ')'
                            ELSE ''
                        END AS DATA_TYPE_FULL,
                    c.IS_NULLABLE,
                    c.COLUMN_DEFAULT,
                    CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END AS IS_PRIMARY_KEY,
                    CASE WHEN ic.object_id IS NOT NULL THEN 1 ELSE 0 END AS IS_IDENTITY,
                    CASE WHEN cc.object_id IS NOT NULL THEN 1 ELSE 0 END AS IS_COMPUTED,
                    cc.definition AS COMPUTED_DEFINITION,
                    CASE WHEN cc.is_persisted = 1 THEN 1 ELSE 0 END AS IS_PERSISTED
                FROM INFORMATION_SCHEMA.COLUMNS c
                LEFT JOIN (
                    SELECT ku.TABLE_NAME, ku.COLUMN_NAME
                    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc
                    INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS ku
                        ON tc.CONSTRAINT_TYPE = 'PRIMARY KEY' 
                        AND tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                ) pk ON c.TABLE_NAME = pk.TABLE_NAME AND c.COLUMN_NAME = pk.COLUMN_NAME
                LEFT JOIN sys.identity_columns ic ON ic.object_id = OBJECT_ID(c.TABLE_SCHEMA + '.' + c.TABLE_NAME) 
                    AND ic.name = c.COLUMN_NAME
                LEFT JOIN sys.computed_columns cc ON cc.object_id = OBJECT_ID(c.TABLE_SCHEMA + '.' + c.TABLE_NAME)
                    AND cc.name = c.COLUMN_NAME
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
                            FieldName = reader.IsDBNull(0) ? string.Empty : reader.GetString(0),
                            SqlDataType = reader.IsDBNull(1) ? string.Empty : reader.GetString(1),
                            IsNullable = reader.IsDBNull(2) ? false : reader.GetString(2).Equals("YES", StringComparison.OrdinalIgnoreCase),
                            DefaultValue = reader.IsDBNull(3) ? null : reader.GetString(3),
                            IsPrimaryKey = reader.IsDBNull(4) ? false : reader.GetInt32(4) == 1,
                            IsAutoIncrement = reader.IsDBNull(5) ? false : reader.GetInt32(5) == 1,
                            IsComputed = reader.IsDBNull(6) ? false : reader.GetInt32(6) == 1,
                            ComputedColumnSql = reader.IsDBNull(7) ? null : reader.GetString(7),
                            IsComputedColumnStored = reader.IsDBNull(8) ? false : reader.GetInt32(8) == 1
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

            using (var connection = new SqlConnection(dbConfig.GetConnectionString()))
            {
                connection.Open();

                var query = @"
                    SELECT 
                        OBJECT_NAME(fk.parent_object_id) AS DependentTable,
                        COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ForeignKeyColumn,
                        OBJECT_NAME(fk.referenced_object_id) AS PrincipalTable,
                        COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS PrincipalKeyColumn,
                        c.is_nullable AS IsNullable
                    FROM sys.foreign_keys fk
                    INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                    INNER JOIN sys.columns c ON fkc.parent_object_id = c.object_id 
                        AND fkc.parent_column_id = c.column_id
                    ORDER BY DependentTable, ForeignKeyColumn";

                using (var command = new SqlCommand(query, connection))
                {
                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            var dependentTable = reader.GetString(0);
                            var foreignKeyColumn = reader.GetString(1);
                            var principalTable = reader.GetString(2);
                            var principalKeyColumn = reader.GetString(3);
                            var isNullable = reader.GetBoolean(4);

                            // Determine relationship type (OneToOne or OneToMany)
                            var relType = DetermineRelationshipType(connection, dependentTable, foreignKeyColumn);

                            relationships.Add(new EntityRelationship
                            {
                                PrincipalEntity = principalTable,
                                PrincipalKey = principalKeyColumn,
                                DependentEntity = dependentTable,
                                ForeignKey = foreignKeyColumn,
                                Type = relType,
                                NavigationType = NavigationType.Bidirectional,
                                IsPrincipalOptional = false,
                                IsDependentOptional = isNullable
                            });
                        }
                    }
                }
            }

            return relationships;
        }

        private RelationshipType DetermineRelationshipType(SqlConnection connection, string tableName, string columnName)
        {
            // Check if the foreign key column is part of a unique constraint or primary key
            var query = @"
                SELECT COUNT(*) 
                FROM sys.indexes i
                INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                WHERE i.is_unique = 1 
                    AND OBJECT_NAME(i.object_id) = @TableName 
                    AND c.name = @ColumnName";

            using (var command = new SqlCommand(query, connection))
            {
                command.Parameters.AddWithValue("@TableName", tableName);
                command.Parameters.AddWithValue("@ColumnName", columnName);

                var count = (int)command.ExecuteScalar();
                return count > 0 ? RelationshipType.OneToOne : RelationshipType.OneToMany;
            }
        }
    }
}

