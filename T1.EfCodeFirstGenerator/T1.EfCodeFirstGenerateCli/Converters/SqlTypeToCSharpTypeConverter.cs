using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace T1.EfCodeFirstGenerateCli.Converters
{
    internal class SqlTypeToCSharpTypeConverter
    {
        private readonly Dictionary<string, Func<string, bool, string>> _customMappings = new Dictionary<string, Func<string, bool, string>>();

        public void RegisterCustomMapping(string sqlType, Func<string, bool, string> converter)
        {
            _customMappings[sqlType.ToLower()] = converter;
        }

        public string ConvertType(string sqlType, bool isNullable)
        {
            var baseType = ExtractBaseType(sqlType);
            
            if (_customMappings.TryGetValue(baseType.ToLower(), out var customConverter))
            {
                return customConverter(sqlType, isNullable);
            }

            var csharpType = GetDefaultCSharpType(baseType);
            
            if (isNullable && IsValueType(csharpType))
            {
                return $"{csharpType}?";
            }

            return csharpType;
        }

        public string GetColumnType(string sqlType)
        {
            return sqlType.ToLower();
        }

        private string ExtractBaseType(string sqlType)
        {
            var match = Regex.Match(sqlType, @"^(\w+)", RegexOptions.IgnoreCase);
            return match.Success ? match.Groups[1].Value : sqlType;
        }

        private string GetDefaultCSharpType(string sqlBaseType)
        {
            switch (sqlBaseType.ToLower())
            {
                case "int":
                case "integer":
                    return "int";
                case "bigint":
                    return "long";
                case "smallint":
                    return "short";
                case "tinyint":
                    return "byte";
                case "bit":
                case "boolean":
                case "bool":
                    return "bool";
                case "decimal":
                case "numeric":
                case "money":
                case "smallmoney":
                    return "decimal";
                case "float":
                case "real":
                    return "double";
                case "date":
                case "datetime":
                case "datetime2":
                case "smalldatetime":
                case "timestamp":
                    return "DateTime";
                case "time":
                    return "TimeSpan";
                case "datetimeoffset":
                    return "DateTimeOffset";
                case "uniqueidentifier":
                case "guid":
                    return "Guid";
                case "binary":
                case "varbinary":
                case "image":
                    return "byte[]";
                case "char":
                case "varchar":
                case "text":
                case "nchar":
                case "nvarchar":
                case "ntext":
                default:
                    return "string";
            }
        }

        private bool IsValueType(string csharpType)
        {
            return csharpType != "string" && csharpType != "byte[]";
        }
    }
}

