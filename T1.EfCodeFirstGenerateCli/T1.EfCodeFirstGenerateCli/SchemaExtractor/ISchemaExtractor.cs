using System.Collections.Generic;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCli.SchemaExtractor
{
    internal interface ISchemaExtractor
    {
        DbSchema ExtractSchema(DbConfig dbConfig);
        List<EntityRelationship> ExtractRelationships(DbConfig dbConfig);
    }
}

