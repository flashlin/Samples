using System;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCli.SchemaExtractor
{
    internal class DatabaseSchemaExtractor
    {
        public static DbSchema CreateDatabaseSchema(DbConfig dbConfig)
        {
            ISchemaExtractor extractor = CreateExtractor(dbConfig.DbType);
            return extractor.ExtractSchema(dbConfig);
        }

        private static ISchemaExtractor CreateExtractor(DbType dbType)
        {
            switch (dbType)
            {
                case DbType.SqlServer:
                    return new SqlServerSchemaExtractor();
                case DbType.MySql:
                    return new MySqlSchemaExtractor();
                default:
                    throw new NotSupportedException($"DbType {dbType} is not supported yet.");
            }
        }
    }
}

