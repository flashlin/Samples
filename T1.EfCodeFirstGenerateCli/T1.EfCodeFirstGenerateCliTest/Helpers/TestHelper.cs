using System;
using System.IO;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCliTest.Helpers
{
    public static class TestHelper
    {
        public static string CreateTempTestDirectory()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), $"EfCodeFirstTest_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempPath);
            return tempPath;
        }

        public static void CleanupDirectory(string directory)
        {
            if (Directory.Exists(directory))
            {
                try
                {
                    Directory.Delete(directory, true);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }

        public static DbSchema CreateTestSchema(string databaseName)
        {
            var schema = new DbSchema
            {
                DatabaseName = databaseName,
                ContextName = databaseName
            };

            // Add Users table
            var usersTable = new TableSchema
            {
                TableName = "Users"
            };
            usersTable.Fields.Add(CreateField("Id", "int", false, null, true, true));
            usersTable.Fields.Add(CreateField("Username", "nvarchar(100)", false, null, false, false));
            usersTable.Fields.Add(CreateField("Email", "nvarchar(255)", true, null, false, false));
            usersTable.Fields.Add(CreateField("CreatedAt", "datetime2", false, "getdate()", false, false));
            usersTable.Fields.Add(CreateField("IsActive", "bit", false, "1", false, false));
            schema.Tables.Add(usersTable);

            // Add Products table
            var productsTable = new TableSchema
            {
                TableName = "Products"
            };
            productsTable.Fields.Add(CreateField("Id", "int", false, null, true, true));
            productsTable.Fields.Add(CreateField("Name", "nvarchar(200)", false, null, false, false));
            productsTable.Fields.Add(CreateField("Price", "decimal(18,2)", false, null, false, false));
            productsTable.Fields.Add(CreateField("Stock", "int", true, null, false, false));
            productsTable.Fields.Add(CreateField("IsActive", "bit", false, "1", false, false));
            schema.Tables.Add(productsTable);

            return schema;
        }

        public static FieldSchema CreateField(
            string fieldName,
            string sqlDataType,
            bool isNullable,
            string? defaultValue,
            bool isPrimaryKey,
            bool isAutoIncrement)
        {
            return new FieldSchema
            {
                FieldName = fieldName,
                SqlDataType = sqlDataType,
                IsNullable = isNullable,
                DefaultValue = defaultValue,
                IsPrimaryKey = isPrimaryKey,
                IsAutoIncrement = isAutoIncrement
            };
        }

        public static FieldSchema CreateComputedField(
            string fieldName,
            string sqlDataType,
            bool isNullable,
            string computedSql,
            bool isStored)
        {
            return new FieldSchema
            {
                FieldName = fieldName,
                SqlDataType = sqlDataType,
                IsNullable = isNullable,
                IsPrimaryKey = false,
                IsAutoIncrement = false,
                IsComputed = true,
                ComputedColumnSql = computedSql,
                IsComputedColumnStored = isStored
            };
        }

        public static bool FileContainsText(string filePath, string text)
        {
            if (!File.Exists(filePath))
                return false;

            var content = File.ReadAllText(filePath);
            return content.Contains(text);
        }

        public static bool ValidateNamespace(string filePath, string expectedNamespace)
        {
            if (!File.Exists(filePath))
                return false;

            var content = File.ReadAllText(filePath);
            return content.Contains($"namespace {expectedNamespace}");
        }

        public static EntityRelationship CreateRelationship(
            string principalEntity,
            string principalKey,
            string dependentEntity,
            string foreignKey,
            RelationshipType type,
            NavigationType navType,
            string? principalNavName = null,
            string? dependentNavName = null)
        {
            return new EntityRelationship
            {
                PrincipalEntity = principalEntity,
                PrincipalKey = principalKey,
                DependentEntity = dependentEntity,
                ForeignKey = foreignKey,
                Type = type,
                NavigationType = navType,
                PrincipalNavigationName = principalNavName,
                DependentNavigationName = dependentNavName
            };
        }
    }
}

