using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Humanizer;
using T1.EfCodeFirstGenerateCli.Common;
using T1.EfCodeFirstGenerateCli.Converters;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCli.CodeGenerator
{
    public class EfCodeGenerator
    {
        private readonly SqlTypeToCSharpTypeConverter _typeConverter;

        public EfCodeGenerator()
        {
            _typeConverter = new SqlTypeToCSharpTypeConverter();
        }

        public EfCodeGenerator(SqlTypeToCSharpTypeConverter typeConverter)
        {
            _typeConverter = typeConverter;
        }

        private class NavigationProperty
        {
            public string Name { get; set; } = string.Empty;
            public string Type { get; set; } = string.Empty;
        }

        private string SanitizeIdentifier(string identifier)
        {
            if (string.IsNullOrEmpty(identifier))
            {
                return identifier;
            }

            // C# reserved keywords
            var reservedKeywords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
                "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else",
                "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for",
                "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock",
                "long", "namespace", "new", "null", "object", "operator", "out", "override", "params",
                "private", "protected", "public", "readonly", "ref", "return", "sbyte", "sealed", "short",
                "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true",
                "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort", "using", "virtual",
                "void", "volatile", "while"
            };

            var sanitized = identifier;
            
            // If identifier starts with a digit, prepend an underscore
            if (char.IsDigit(sanitized[0]))
            {
                sanitized = "_" + sanitized;
            }
            
            // Replace other invalid characters with underscore
            sanitized = Regex.Replace(sanitized, @"[^\w]", "_");
            
            // If after sanitization it starts with a digit, prepend underscore
            if (char.IsDigit(sanitized[0]))
            {
                sanitized = "_" + sanitized;
            }
            
            // If it's a reserved keyword, prepend underscore
            if (reservedKeywords.Contains(sanitized))
            {
                sanitized = "_" + sanitized;
            }
            
            return sanitized;
        }

        private string ToPascalCase(string identifier)
        {
            if (string.IsNullOrEmpty(identifier))
                return identifier;
            
            var sanitized = SanitizeIdentifier(identifier);
            
            if (char.IsLower(sanitized[0]))
            {
                return char.ToUpper(sanitized[0]) + sanitized.Substring(1);
            }
            
            return sanitized;
        }

        private string ToPlural(string singular)
        {
            var pluralized = singular.Pluralize();
            
            if (pluralized == singular)
            {
                return singular;
            }
            
            return pluralized;
        }

        public Dictionary<string, string> GenerateCodeFirstFromSchema(DbSchema dbSchema, string targetNamespace)
        {
            var generatedFiles = new Dictionary<string, string>();

            var dbContextCode = GenerateDbContext(dbSchema, targetNamespace);
            generatedFiles[$"{dbSchema.ContextName}/{dbSchema.ContextName}DbContext.cs"] = dbContextCode;

            foreach (var table in dbSchema.Tables)
            {
                var entityCode = GenerateEntity(table, targetNamespace, dbSchema.Relationships);
                generatedFiles[$"{dbSchema.ContextName}/Entities/{table.TableName}Entity.cs"] = entityCode;

                var configCode = GenerateEntityConfiguration(table, targetNamespace, dbSchema.Relationships);
                generatedFiles[$"{dbSchema.ContextName}/Configurations/{table.TableName}EntityConfiguration.cs"] = configCode;
            }

            return generatedFiles;
        }

        private string GenerateDbContext(DbSchema dbSchema, string targetNamespace)
        {
            var output = new IndentStringBuilder();

            output.WriteLine($"// This file is auto-generated. Do not modify manually.");
            output.WriteLine("using Microsoft.EntityFrameworkCore;");
            output.WriteLine($"using {targetNamespace}.Entities;");
            output.WriteLine($"using {targetNamespace}.Configurations;");
            output.WriteLine();
            output.WriteLine($"namespace {targetNamespace}");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public partial class {dbSchema.ContextName}DbContext : DbContext");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public {dbSchema.ContextName}DbContext(DbContextOptions<{dbSchema.ContextName}DbContext> options)");
            output.Indent++;
            output.WriteLine(": base(options)");
            output.Indent--;
            output.WriteLine("{");
            output.WriteLine("}");
            output.WriteLine();
            output.WriteLine($"public {dbSchema.ContextName}DbContext()");
            output.WriteLine("{");
            output.WriteLine("}");
            output.WriteLine();

            foreach (var table in dbSchema.Tables)
            {
                var propertyName = ToPascalCase(table.TableName);
                output.WriteLine($"public DbSet<{table.TableName}Entity> {propertyName} {{ get; set; }}");
            }

            output.WriteLine();
            output.WriteLine("protected override void OnModelCreating(ModelBuilder modelBuilder)");
            output.WriteLine("{");
            output.Indent++;

            foreach (var table in dbSchema.Tables)
            {
                output.WriteLine($"modelBuilder.ApplyConfiguration(new {table.TableName}EntityConfiguration());");
            }

            output.Indent--;
            output.WriteLine("}");

            output.Indent--;
            output.WriteLine("}");

            output.Indent--;
            output.WriteLine("}");

            return output.ToString();
        }

        private string GenerateEntity(TableSchema table, string targetNamespace, List<EntityRelationship> relationships)
        {
            var output = new IndentStringBuilder();
            
            WriteEntityFileHeader(output);
            WriteEntityNamespace(output, targetNamespace);
            WriteEntityClassDeclaration(output, table.TableName);
            WriteEntityProperties(output, table.Fields);
            WriteEntityNavigationProperties(output, table.TableName, relationships);
            CloseEntityClass(output);
            
            return output.ToString();
        }

        private void WriteEntityFileHeader(IndentStringBuilder output)
        {
            output.WriteLine($"// This file is auto-generated. Do not modify manually.");
            output.WriteLine("using System;");
            output.WriteLine("using System.Collections.Generic;");
            output.WriteLine();
        }

        private void WriteEntityNamespace(IndentStringBuilder output, string targetNamespace)
        {
            output.WriteLine($"namespace {targetNamespace}.Entities");
            output.WriteLine("{");
            output.Indent++;
        }

        private void WriteEntityClassDeclaration(IndentStringBuilder output, string tableName)
        {
            output.WriteLine($"public partial class {tableName}Entity");
            output.WriteLine("{");
            output.Indent++;
        }

        private void WriteEntityProperties(IndentStringBuilder output, List<FieldSchema> fields)
        {
            foreach (var field in fields)
            {
                var csharpType = _typeConverter.ConvertType(field.SqlDataType, field.IsNullable);
                var requiredModifier = IsNonNullableReferenceType(csharpType) ? "required " : "";
                var propertyName = ToPascalCase(field.FieldName);
                output.WriteLine($"public {requiredModifier}{csharpType} {propertyName} {{ get; set; }}");
            }
        }

        private void WriteEntityNavigationProperties(IndentStringBuilder output, string tableName, List<EntityRelationship> relationships)
        {
            var navProps = GetNavigationPropertiesForEntity(tableName, relationships);
            if (navProps.Count == 0)
                return;
            
            output.WriteLine();
            output.WriteLine("// Navigation properties");
            
            foreach (var navProp in navProps)
            {
                if (navProp.Type.StartsWith("ICollection"))
                {
                    output.WriteLine($"public {navProp.Type} {navProp.Name} {{ get; set; }} = new List<{ExtractGenericType(navProp.Type)}>();");
                }
                else
                {
                    output.WriteLine($"public {navProp.Type}? {navProp.Name} {{ get; set; }}");
                }
            }
        }

        private void CloseEntityClass(IndentStringBuilder output)
        {
            output.Indent--;
            output.WriteLine("}");
            output.Indent--;
            output.WriteLine("}");
        }

        private bool IsNonNullableReferenceType(string csharpType)
        {
            return (csharpType == "string" || csharpType == "byte[]") && !csharpType.EndsWith("?");
        }

        private string GenerateEntityConfiguration(TableSchema table, string targetNamespace, List<EntityRelationship> relationships)
        {
            var output = new IndentStringBuilder();

            WriteConfigurationFileHeader(output, targetNamespace);
            WriteConfigurationNamespace(output, targetNamespace);
            WriteConfigurationClassDeclaration(output, table.TableName);
            WriteConfigureMethodStart(output, table.TableName);
            WriteTableConfiguration(output, table.TableName);
            WritePrimaryKeyConfiguration(output, table);
            WritePropertiesConfiguration(output, table);
            GenerateRelationshipConfigurations(output, table.TableName, relationships);
            WriteConfigureMethodEnd(output, table.TableName);
            CloseConfigurationClass(output);

            return output.ToString();
        }

        private void WriteConfigurationFileHeader(IndentStringBuilder output, string targetNamespace)
        {
            output.WriteLine($"// This file is auto-generated. Do not modify manually.");
            output.WriteLine("using Microsoft.EntityFrameworkCore;");
            output.WriteLine("using Microsoft.EntityFrameworkCore.Metadata.Builders;");
            output.WriteLine($"using {targetNamespace}.Entities;");
            output.WriteLine();
        }

        private void WriteConfigurationNamespace(IndentStringBuilder output, string targetNamespace)
        {
            output.WriteLine($"namespace {targetNamespace}.Configurations");
            output.WriteLine("{");
            output.Indent++;
        }

        private void WriteConfigurationClassDeclaration(IndentStringBuilder output, string tableName)
        {
            output.WriteLine($"public partial class {tableName}EntityConfiguration : IEntityTypeConfiguration<{tableName}Entity>");
            output.WriteLine("{");
            output.Indent++;
        }

        private void WriteConfigureMethodStart(IndentStringBuilder output, string tableName)
        {
            output.WriteLine($"public void Configure(EntityTypeBuilder<{tableName}Entity> builder)");
            output.WriteLine("{");
            output.Indent++;
        }

        private void WriteTableConfiguration(IndentStringBuilder output, string tableName)
        {
            output.WriteLine($"builder.ToTable(\"{tableName}\");");
            output.WriteLine();
        }

        private void WritePrimaryKeyConfiguration(IndentStringBuilder output, TableSchema table)
        {
            var primaryKeys = table.Fields.Where(f => f.IsPrimaryKey).ToList();
            if (primaryKeys.Count == 1)
            {
                var pk = primaryKeys[0];
                var pkPropertyName = ToPascalCase(pk.FieldName);
                output.WriteLine($"builder.HasKey(x => x.{pkPropertyName});");
                
                // Check if primary key needs ValueGeneratedNever
                if (!pk.IsAutoIncrement && HasOtherAutoIncrementFields(table, pk))
                {
                    output.WriteLine();
                    output.WriteLine($"builder.Property(x => x.{pkPropertyName})");
                    output.Indent++;
                    output.WriteLine(".ValueGeneratedNever();");
                    output.Indent--;
                }
            }
            else if (primaryKeys.Count > 1)
            {
                var pkFields = string.Join(", ", primaryKeys.Select(pk => $"x.{ToPascalCase(pk.FieldName)}"));
                output.WriteLine($"builder.HasKey(x => new {{ {pkFields} }});");
                
                // Check each primary key field
                foreach (var pk in primaryKeys)
                {
                    if (!pk.IsAutoIncrement && HasOtherAutoIncrementFields(table, pk))
                    {
                        var pkPropertyName = ToPascalCase(pk.FieldName);
                        output.WriteLine();
                        output.WriteLine($"builder.Property(x => x.{pkPropertyName})");
                        output.Indent++;
                        output.WriteLine(".ValueGeneratedNever();");
                        output.Indent--;
                    }
                }
            }
            else
            {
                // No primary key - this is a keyless entity
                output.WriteLine("builder.HasNoKey();");
            }

            output.WriteLine();
        }

        private void WritePropertiesConfiguration(IndentStringBuilder output, TableSchema table)
        {
            foreach (var field in table.Fields)
            {
                GeneratePropertyConfiguration(output, field);
            }
        }

        private void WriteConfigureMethodEnd(IndentStringBuilder output, string tableName)
        {
            output.WriteLine($"ConfigureCustomProperties(builder);");
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
            output.WriteLine($"partial void ConfigureCustomProperties(EntityTypeBuilder<{tableName}Entity> builder);");
        }

        private void CloseConfigurationClass(IndentStringBuilder output)
        {
            output.Indent--;
            output.WriteLine("}");
            output.Indent--;
            output.WriteLine("}");
        }

        private void GeneratePropertyConfiguration(IndentStringBuilder output, FieldSchema field)
        {
            var propertyName = ToPascalCase(field.FieldName);
            
            output.WriteLine($"builder.Property(x => x.{propertyName})");
            output.Indent++;
            
            // Check if this is a computed column
            if (field.IsComputed && !string.IsNullOrEmpty(field.ComputedColumnSql))
            {
                var computedSql = field.ComputedColumnSql.Trim();
                // Escape double quotes in SQL expression
                computedSql = computedSql.Replace("\"", "\\\"");
                var storedParam = field.IsComputedColumnStored ? "stored: true" : "stored: false";
                
                output.WriteLine($".HasComputedColumnSql(\"{computedSql}\", {storedParam})");
                
                // Computed columns don't need other configurations
                output.Indent--;
                output.WriteLine(";");
                output.WriteLine();
                return;
            }
            
            var columnType = _typeConverter.GetColumnType(field.SqlDataType);
            
            // Check if this is timestamp/rowversion
            var baseType = ExtractBaseType(field.SqlDataType);
            var isTimestamp = baseType.Equals("timestamp", StringComparison.OrdinalIgnoreCase) ||
                              baseType.Equals("rowversion", StringComparison.OrdinalIgnoreCase);
            
            if (isTimestamp)
            {
                // timestamp/rowversion special handling
                output.WriteLine($".HasColumnType(\"{columnType}\")");
                output.WriteLine(".IsRowVersion()");
                output.WriteLine(".ValueGeneratedOnAddOrUpdate()");
                
                output.Indent--;
                output.WriteLine(";");
                output.WriteLine();
                return;
            }
            
            // If property name was sanitized, add HasColumnName to map to original column
            if (propertyName != field.FieldName)
            {
                output.WriteLine($".HasColumnName(\"{field.FieldName}\")");
            }
            
            output.WriteLine($".HasColumnType(\"{columnType}\")");

            if (field.IsAutoIncrement)
            {
                output.WriteLine(".ValueGeneratedOnAdd()");
            }

            if (!field.IsNullable)
            {
                output.WriteLine(".IsRequired()");
            }

            var maxLength = ExtractMaxLength(field.SqlDataType);
            if (maxLength.HasValue && maxLength.Value > 0)
            {
                output.WriteLine($".HasMaxLength({maxLength.Value})");
            }

            if (!string.IsNullOrEmpty(field.DefaultValue))
            {
                var (isSqlFunction, defaultValue) = FormatDefaultValue(field.DefaultValue!, field.SqlDataType);
                if (!string.IsNullOrEmpty(defaultValue))
                {
                    var method = isSqlFunction ? "HasDefaultValueSql" : "HasDefaultValue";
                    output.WriteLine($".{method}({defaultValue})");
                }
            }

            output.Indent--;
            output.WriteLine(";");
            output.WriteLine();
        }

        private int? ExtractMaxLength(string sqlDataType)
        {
            var match = Regex.Match(sqlDataType, @"\((\d+)\)", RegexOptions.IgnoreCase);
            if (match.Success && int.TryParse(match.Groups[1].Value, out var length))
            {
                return length;
            }
            return null;
        }

        private (bool isSqlFunction, string value) FormatDefaultValue(string defaultValue, string sqlDataType)
        {
            // Check if it's a SQL function or expression (contains parentheses or special keywords)
            if (IsSqlFunctionOrExpression(defaultValue))
            {
                return (true, $"\"{defaultValue.Trim()}\"");
            }

            defaultValue = defaultValue.Trim('(', ')', '\'', '"');
            
            var baseType = Regex.Match(sqlDataType, @"^(\w+)").Groups[1].Value.ToLower();
            
            switch (baseType)
            {
                case "decimal":
                case "numeric":
                    // Add 'm' suffix for decimal types
                    if (decimal.TryParse(defaultValue, out var decimalValue))
                    {
                        return (false, $"{decimalValue}m");
                    }
                    return (false, defaultValue);
                    
                case "float":
                case "real":
                    // Add 'f' suffix for float types
                    if (double.TryParse(defaultValue, out var floatValue))
                    {
                        return (false, $"{floatValue}f");
                    }
                    return (false, defaultValue);
                    
                case "bigint":
                    // Add 'L' suffix for long types
                    if (long.TryParse(defaultValue, out var longValue))
                    {
                        return (false, $"{longValue}L");
                    }
                    return (false, defaultValue);
                    
                case "tinyint":
                    // Add explicit cast for byte types
                    if (byte.TryParse(defaultValue, out var byteValue))
                    {
                        return (false, $"(byte){byteValue}");
                    }
                    return (false, defaultValue);
                    
                case "smallint":
                    // Add explicit cast for short types
                    if (short.TryParse(defaultValue, out var shortValue))
                    {
                        return (false, $"(short){shortValue}");
                    }
                    return (false, defaultValue);
                    
                case "int":
                    return (false, defaultValue);
                    
                case "bit":
                case "boolean":
                    return (false, defaultValue.ToLower() == "1" || defaultValue.ToLower() == "true" ? "true" : "false");
                    
                case "varchar":
                case "nvarchar":
                case "char":
                case "nchar":
                case "text":
                    return (false, $"\"{defaultValue}\"");
                    
                default:
                    return (false, string.Empty);
            }
        }

        private bool IsSqlFunctionOrExpression(string defaultValue)
        {
            var trimmedValue = defaultValue.Trim();
            
            // Check if it contains function call pattern
            if (Regex.IsMatch(trimmedValue, @"\w+\s*\(.*\)", RegexOptions.IgnoreCase))
            {
                return true;
            }
            
            // Check for common SQL functions
            var sqlFunctions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "getdate", "getutcdate", "newid", "current_timestamp", "current_date", 
                "current_time", "now", "uuid", "user", "session_user", "system_user"
            };
            
            var functionName = Regex.Match(trimmedValue, @"^(\w+)", RegexOptions.IgnoreCase).Groups[1].Value;
            if (sqlFunctions.Contains(functionName))
            {
                return true;
            }
            
            return false;
        }

        private bool HasOtherAutoIncrementFields(TableSchema table, FieldSchema currentField)
        {
            return table.Fields.Any(f => 
                f.FieldName != currentField.FieldName && 
                f.IsAutoIncrement);
        }

        private string ExtractBaseType(string sqlDataType)
        {
            var match = Regex.Match(sqlDataType, @"^(\w+)", RegexOptions.IgnoreCase);
            return match.Success ? match.Groups[1].Value : sqlDataType;
        }

        private List<NavigationProperty> GetNavigationPropertiesForEntity(
            string tableName, 
            List<EntityRelationship> relationships)
        {
            var navProps = new List<NavigationProperty>();
            
            var principalRels = relationships.Where(r => r.PrincipalEntity == tableName).ToList();
            var grouped = principalRels.GroupBy(r => r.DependentEntity);
            
            foreach (var group in grouped)
            {
                if (group.Count() > 1)
                {
                    foreach (var rel in group)
                    {
                        var navName = rel.PrincipalNavigationName ?? GenerateUniqueNavigationName(rel, rel.ForeignKey);
                        var navType = (rel.Type == RelationshipType.OneToOne) 
                            ? $"{rel.DependentEntity}Entity"
                            : $"ICollection<{rel.DependentEntity}Entity>";
                        navProps.Add(new NavigationProperty { Name = navName, Type = navType });
                    }
                }
                else
                {
                    var rel = group.First();
                    var navName = rel.PrincipalNavigationName ?? GetDefaultNavigationName(rel);
                    var navType = (rel.Type == RelationshipType.OneToOne) 
                        ? $"{rel.DependentEntity}Entity"
                        : $"ICollection<{rel.DependentEntity}Entity>";
                    navProps.Add(new NavigationProperty { Name = navName, Type = navType });
                }
            }
            
            var dependentRels = relationships.Where(r => r.DependentEntity == tableName && r.NavigationType == NavigationType.Bidirectional).ToList();
            var dependentGrouped = dependentRels.GroupBy(r => r.PrincipalEntity);
            
            foreach (var group in dependentGrouped)
            {
                if (group.Count() > 1)
                {
                    foreach (var rel in group)
                    {
                        var navName = rel.DependentNavigationName ?? GetDependentNavigationName(rel, relationships);
                        var navType = $"{rel.PrincipalEntity}Entity";
                        navProps.Add(new NavigationProperty { Name = navName, Type = navType });
                    }
                }
                else
                {
                    var rel = group.First();
                    var navName = rel.DependentNavigationName ?? rel.PrincipalEntity;
                    var navType = $"{rel.PrincipalEntity}Entity";
                    navProps.Add(new NavigationProperty { Name = navName, Type = navType });
                }
            }
            
            return navProps;
        }

        private string GetDefaultNavigationName(EntityRelationship rel)
        {
            if (rel.Type == RelationshipType.OneToMany || rel.Type == RelationshipType.ManyToOne)
            {
                return ToPlural(rel.DependentEntity);
            }
            return rel.DependentEntity;
        }

        private string GenerateUniqueNavigationName(EntityRelationship rel, string foreignKeyName)
        {
            var fkName = ToPascalCase(foreignKeyName);
            if (fkName.EndsWith("Id"))
            {
                fkName = fkName.Substring(0, fkName.Length - 2);
            }
            
            if (rel.Type == RelationshipType.OneToMany || rel.Type == RelationshipType.ManyToOne)
            {
                return $"{ToPlural(rel.DependentEntity)}By{fkName}";
            }
            return $"{rel.DependentEntity}By{fkName}";
        }

        private string ExtractGenericType(string type)
        {
            // Extract "OrderEntity" from "ICollection<OrderEntity>"
            var match = Regex.Match(type, @"<(.+)>");
            return match.Success ? match.Groups[1].Value : type;
        }

        private string GetPrincipalNavigationName(EntityRelationship rel, List<EntityRelationship> allRelationships)
        {
            if (!string.IsNullOrEmpty(rel.PrincipalNavigationName))
            {
                return rel.PrincipalNavigationName;
            }
            
            var sameTargetRels = allRelationships
                .Where(r => r.PrincipalEntity == rel.PrincipalEntity && r.DependentEntity == rel.DependentEntity)
                .ToList();
            
            if (sameTargetRels.Count > 1)
            {
                return GenerateUniqueNavigationName(rel, rel.ForeignKey);
            }
            
            return GetDefaultNavigationName(rel);
        }

        private string GetDependentNavigationName(EntityRelationship rel, List<EntityRelationship> allRelationships)
        {
            if (!string.IsNullOrEmpty(rel.DependentNavigationName))
            {
                return rel.DependentNavigationName;
            }
            
            var samePrincipalRels = allRelationships
                .Where(r => r.DependentEntity == rel.DependentEntity && r.PrincipalEntity == rel.PrincipalEntity)
                .ToList();
            
            if (samePrincipalRels.Count > 1)
            {
                var fkName = ToPascalCase(rel.ForeignKey);
                if (fkName.EndsWith("Id"))
                {
                    fkName = fkName.Substring(0, fkName.Length - 2);
                }
                return $"{rel.PrincipalEntity}By{fkName}";
            }
            
            return rel.PrincipalEntity;
        }

        private void GenerateRelationshipConfigurations(
            IndentStringBuilder output, 
            string tableName, 
            List<EntityRelationship> relationships)
        {
            var relevantRels = relationships.Where(r => 
                r.DependentEntity == tableName).ToList();
            
            if (relevantRels.Count == 0)
                return;
            
            output.WriteLine();
            output.WriteLine("// Relationship configurations");
            
            foreach (var rel in relevantRels)
            {
                GenerateSingleRelationship(output, tableName, rel, relationships);
            }
        }

        private void GenerateSingleRelationship(
            IndentStringBuilder output, 
            string dependentTable, 
            EntityRelationship rel,
            List<EntityRelationship> allRelationships)
        {
            var fkProp = ToPascalCase(rel.ForeignKey);
            
            if (rel.Type == RelationshipType.OneToOne)
            {
                if (rel.NavigationType == NavigationType.Bidirectional)
                {
                    WriteOneToOneBidirectionalRelationship(output, dependentTable, rel, fkProp, allRelationships);
                }
                else
                {
                    WriteOneToOneUnidirectionalRelationship(output, dependentTable, rel, fkProp);
                }
            }
            else
            {
                if (rel.NavigationType == NavigationType.Bidirectional)
                {
                    WriteOneToManyBidirectionalRelationship(output, rel, fkProp, allRelationships);
                }
                else
                {
                    WriteOneToManyUnidirectionalRelationship(output, rel, fkProp);
                }
            }
            
            output.WriteLine();
        }

        private void WriteOneToOneBidirectionalRelationship(
            IndentStringBuilder output,
            string dependentTable,
            EntityRelationship rel,
            string fkProp,
            List<EntityRelationship> allRelationships)
        {
            var depNav = GetDependentNavigationName(rel, allRelationships);
            var prinNav = GetPrincipalNavigationName(rel, allRelationships);
            output.WriteLine($"builder.HasOne(x => x.{depNav})");
            output.Indent++;
            output.WriteLine($".WithOne(x => x.{prinNav})");
            WriteForeignKeyWithOptional(output, $"<{dependentTable}Entity>", fkProp, rel.IsDependentOptional);
            output.Indent--;
        }

        private void WriteOneToOneUnidirectionalRelationship(
            IndentStringBuilder output,
            string dependentTable,
            EntityRelationship rel,
            string fkProp)
        {
            output.WriteLine($"builder.HasOne<{rel.PrincipalEntity}Entity>()");
            output.Indent++;
            output.WriteLine(".WithOne()");
            WriteForeignKeyWithOptional(output, $"<{dependentTable}Entity>", fkProp, rel.IsDependentOptional);
            output.Indent--;
        }

        private void WriteOneToManyBidirectionalRelationship(
            IndentStringBuilder output,
            EntityRelationship rel,
            string fkProp,
            List<EntityRelationship> allRelationships)
        {
            var depNav = GetDependentNavigationName(rel, allRelationships);
            var prinNav = GetPrincipalNavigationName(rel, allRelationships);
            output.WriteLine($"builder.HasOne(x => x.{depNav})");
            output.Indent++;
            output.WriteLine($".WithMany(x => x.{prinNav})");
            WriteForeignKeyWithOptional(output, "", fkProp, rel.IsDependentOptional);
            output.Indent--;
        }

        private void WriteOneToManyUnidirectionalRelationship(
            IndentStringBuilder output,
            EntityRelationship rel,
            string fkProp)
        {
            output.WriteLine($"builder.HasOne<{rel.PrincipalEntity}Entity>()");
            output.Indent++;
            output.WriteLine(".WithMany()");
            WriteForeignKeyWithOptional(output, "", fkProp, rel.IsDependentOptional);
            output.Indent--;
        }

        private void WriteForeignKeyWithOptional(
            IndentStringBuilder output,
            string entityTypeParameter,
            string fkProp,
            bool isOptional)
        {
            if (isOptional)
            {
                output.WriteLine($".HasForeignKey{entityTypeParameter}(x => x.{fkProp})");
                output.WriteLine(".IsRequired(false);");
            }
            else
            {
                output.WriteLine($".HasForeignKey{entityTypeParameter}(x => x.{fkProp});");
            }
        }
    }
}

