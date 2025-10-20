using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using T1.EfCodeFirstGenerator.Common;
using T1.EfCodeFirstGenerator.Converters;
using T1.EfCodeFirstGenerator.Models;

namespace T1.EfCodeFirstGenerator.CodeGenerator
{
    internal class EfCodeGenerator
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

        public Dictionary<string, string> GenerateCodeFirstFromSchema(DbSchema dbSchema, string targetNamespace)
        {
            var generatedFiles = new Dictionary<string, string>();

            var dbContextCode = GenerateDbContext(dbSchema, targetNamespace);
            generatedFiles[$"{dbSchema.DatabaseName}DbContext.cs"] = dbContextCode;

            foreach (var table in dbSchema.Tables)
            {
                var entityCode = GenerateEntity(table, targetNamespace);
                generatedFiles[$"{table.TableName}Entity.cs"] = entityCode;

                var configCode = GenerateEntityConfiguration(table, targetNamespace);
                generatedFiles[$"{table.TableName}EntityConfiguration.cs"] = configCode;
            }

            return generatedFiles;
        }

        private string GenerateDbContext(DbSchema dbSchema, string targetNamespace)
        {
            var output = new IndentStringBuilder();

            output.WriteLine("using Microsoft.EntityFrameworkCore;");
            output.WriteLine();
            output.WriteLine($"namespace {targetNamespace}");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public partial class {dbSchema.DatabaseName}DbContext : DbContext");
            output.WriteLine("{");
            output.Indent++;

            foreach (var table in dbSchema.Tables)
            {
                output.WriteLine($"public DbSet<{table.TableName}Entity> {table.TableName} {{ get; set; }}");
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

            output.WriteLine("using System;");
            output.WriteLine();
            output.WriteLine($"namespace {targetNamespace}");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public class {table.TableName}Entity");
            output.WriteLine("{");
            output.Indent++;

            foreach (var field in table.Fields)
            {
                var csharpType = _typeConverter.ConvertType(field.SqlDataType, field.IsNullable);
                var requiredModifier = IsNonNullableReferenceType(csharpType) ? "required " : "";
                output.WriteLine($"public {requiredModifier}{csharpType} {field.FieldName} {{ get; set; }}");
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

            output.WriteLine("using Microsoft.EntityFrameworkCore;");
            output.WriteLine("using Microsoft.EntityFrameworkCore.Metadata.Builders;");
            output.WriteLine();
            output.WriteLine($"namespace {targetNamespace}");
            output.WriteLine("{");
            output.Indent++;

            output.WriteLine($"public class {table.TableName}EntityConfiguration : IEntityTypeConfiguration<{table.TableName}Entity>");
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
                output.WriteLine($"builder.HasKey(x => x.{pk.FieldName});");
            }
            else if (primaryKeys.Count > 1)
            {
                var pkFields = string.Join(", ", primaryKeys.Select(pk => $"x.{pk.FieldName}"));
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
            
            output.WriteLine($"builder.Property(x => x.{field.FieldName})");
            output.Indent++;
            output.WriteLine($".HasColumnType(\"{columnType}\")");

            if (field.IsPrimaryKey)
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
                var defaultValue = FormatDefaultValue(field.DefaultValue!, field.SqlDataType);
                if (!string.IsNullOrEmpty(defaultValue))
                {
                    output.WriteLine($".HasDefaultValue({defaultValue})");
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

        private string FormatDefaultValue(string defaultValue, string sqlDataType)
        {
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
                    return defaultValue;
                case "bit":
                case "boolean":
                    return defaultValue.ToLower() == "1" || defaultValue.ToLower() == "true" ? "true" : "false";
                case "varchar":
                case "nvarchar":
                case "char":
                case "nchar":
                case "text":
                    return $"\"{defaultValue}\"";
                default:
                    return string.Empty;
            }
        }
    }
}

