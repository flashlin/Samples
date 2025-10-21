using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
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

        public Dictionary<string, string> GenerateCodeFirstFromSchema(DbSchema dbSchema, string targetNamespace)
        {
            var generatedFiles = new Dictionary<string, string>();

            var dbContextCode = GenerateDbContext(dbSchema, targetNamespace);
            generatedFiles[$"{dbSchema.DatabaseName}/{dbSchema.DatabaseName}DbContext.cs"] = dbContextCode;

            foreach (var table in dbSchema.Tables)
            {
                var entityCode = GenerateEntity(table, targetNamespace);
                generatedFiles[$"{dbSchema.DatabaseName}/Entities/{table.TableName}Entity.cs"] = entityCode;

                var configCode = GenerateEntityConfiguration(table, targetNamespace);
                generatedFiles[$"{dbSchema.DatabaseName}/Configurations/{table.TableName}EntityConfiguration.cs"] = configCode;
            }

            return generatedFiles;
        }

        private string GenerateDbContext(DbSchema dbSchema, string targetNamespace)
        {
            var output = new IndentStringBuilder();

            output.WriteLine($"// This file is auto-generated. Do not modify manually.");
            output.WriteLine($"// Generated at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            output.WriteLine("using Microsoft.EntityFrameworkCore;");
            output.WriteLine($"using {targetNamespace}.Entities;");
            output.WriteLine($"using {targetNamespace}.Configurations;");
            output.WriteLine();
            output.WriteLine($"namespace {targetNamespace}");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public partial class {dbSchema.DatabaseName}DbContext : DbContext");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public {dbSchema.DatabaseName}DbContext(DbContextOptions<{dbSchema.DatabaseName}DbContext> options)");
            output.Indent++;
            output.WriteLine(": base(options)");
            output.Indent--;
            output.WriteLine("{");
            output.WriteLine("}");
            output.WriteLine();
            output.WriteLine($"public {dbSchema.DatabaseName}DbContext()");
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

        private string GenerateEntity(TableSchema table, string targetNamespace)
        {
            var output = new IndentStringBuilder();
            
            output.WriteLine($"// This file is auto-generated. Do not modify manually.");
            output.WriteLine($"// Generated at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            output.WriteLine("using System;");
            output.WriteLine();
            output.WriteLine($"namespace {targetNamespace}.Entities");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public partial class {table.TableName}Entity");
            output.WriteLine("{");
            output.Indent++;

            foreach (var field in table.Fields)
            {
                var csharpType = _typeConverter.ConvertType(field.SqlDataType, field.IsNullable);
                var requiredModifier = IsNonNullableReferenceType(csharpType) ? "required " : "";
                var propertyName = ToPascalCase(field.FieldName);
                output.WriteLine($"public {requiredModifier}{csharpType} {propertyName} {{ get; set; }}");
            }

            output.Indent--;
            output.WriteLine("}");

            output.Indent--;
            output.WriteLine("}");

            return output.ToString();
        }

        private bool IsNonNullableReferenceType(string csharpType)
        {
            return (csharpType == "string" || csharpType == "byte[]") && !csharpType.EndsWith("?");
        }

        private string GenerateEntityConfiguration(TableSchema table, string targetNamespace)
        {
            var output = new IndentStringBuilder();

            output.WriteLine($"// This file is auto-generated. Do not modify manually.");
            output.WriteLine($"// Generated at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            output.WriteLine("using Microsoft.EntityFrameworkCore;");
            output.WriteLine("using Microsoft.EntityFrameworkCore.Metadata.Builders;");
            output.WriteLine($"using {targetNamespace}.Entities;");
            output.WriteLine();
            output.WriteLine($"namespace {targetNamespace}.Configurations");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public partial class {table.TableName}EntityConfiguration : IEntityTypeConfiguration<{table.TableName}Entity>");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public void Configure(EntityTypeBuilder<{table.TableName}Entity> builder)");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"builder.ToTable(\"{table.TableName}\");");
            output.WriteLine();

            var primaryKeys = table.Fields.Where(f => f.IsPrimaryKey).ToList();
            if (primaryKeys.Count == 1)
            {
                var pk = primaryKeys[0];
                var pkPropertyName = ToPascalCase(pk.FieldName);
                output.WriteLine($"builder.HasKey(x => x.{pkPropertyName});");
            }
            else if (primaryKeys.Count > 1)
            {
                var pkFields = string.Join(", ", primaryKeys.Select(pk => $"x.{ToPascalCase(pk.FieldName)}"));
                output.WriteLine($"builder.HasKey(x => new {{ {pkFields} }});");
            }

            output.WriteLine();

            foreach (var field in table.Fields)
            {
                GeneratePropertyConfiguration(output, field);
            }

            output.Indent--;
            output.WriteLine("}");

            output.Indent--;
            output.WriteLine("}");

            output.Indent--;
            output.WriteLine("}");

            return output.ToString();
        }

        private void GeneratePropertyConfiguration(IndentStringBuilder output, FieldSchema field)
        {
            var columnType = _typeConverter.GetColumnType(field.SqlDataType);
            var propertyName = ToPascalCase(field.FieldName);
            
            output.WriteLine($"builder.Property(x => x.{propertyName})");
            output.Indent++;
            
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
                case "int":
                case "bigint":
                case "smallint":
                case "tinyint":
                case "decimal":
                case "numeric":
                case "float":
                case "real":
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
    }
}

