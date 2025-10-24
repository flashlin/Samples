using System;
using System.Collections.Generic;
using System.Linq;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCli.SchemaExtractor
{
    internal class DatabaseSchemaExtractor
    {
        public static DbSchema CreateDatabaseSchema(DbConfig dbConfig)
        {
            ISchemaExtractor extractor = CreateExtractor(dbConfig.DbType);
            var dbSchema = extractor.ExtractSchema(dbConfig);
            
            // Extract relationships from database foreign keys
            var dbRelationships = extractor.ExtractRelationships(dbConfig);
            
            // Add relationships from .db file (Mermaid definitions)
            var mermaidRelationships = dbConfig.Relationships ?? new List<EntityRelationship>();
            
            // Merge: Mermaid relationships have priority, filter out duplicates from database
            var mergedRelationships = MergeRelationships(dbRelationships, mermaidRelationships);
            dbSchema.Relationships.AddRange(mergedRelationships);
            
            return dbSchema;
        }
        
        private static List<EntityRelationship> MergeRelationships(
            List<EntityRelationship> dbRelationships,
            List<EntityRelationship> mermaidRelationships)
        {
            var result = new List<EntityRelationship>();
            
            // Add all Mermaid relationships first (they have priority)
            result.AddRange(mermaidRelationships);
            
            // Add database relationships that don't conflict with Mermaid definitions
            foreach (var dbRel in dbRelationships)
            {
                // Check if this relationship already exists in Mermaid definitions
                var isDuplicate = mermaidRelationships.Any(mRel =>
                    mRel.PrincipalEntity == dbRel.PrincipalEntity &&
                    mRel.DependentEntity == dbRel.DependentEntity &&
                    mRel.ForeignKey == dbRel.ForeignKey);
                
                if (!isDuplicate)
                {
                    result.Add(dbRel);
                }
            }
            
            return result;
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

