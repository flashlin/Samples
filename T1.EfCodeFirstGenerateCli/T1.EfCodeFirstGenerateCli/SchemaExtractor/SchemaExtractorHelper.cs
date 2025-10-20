using System.Collections.Generic;
using System.Data.Common;

namespace T1.EfCodeFirstGenerateCli.SchemaExtractor
{
    internal static class SchemaExtractorHelper
    {
        public static List<string> GetTables(DbConnection connection, string databaseName)
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
    }
}

