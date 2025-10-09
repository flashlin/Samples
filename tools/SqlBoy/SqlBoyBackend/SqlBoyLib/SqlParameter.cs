namespace SqlBoyLib;

public class SqlParameter
{
    public string Name { get; set; } = string.Empty;
    public string SqlType { get; set; } = string.Empty;
    public object? Value { get; set; }

    public static string GetSqlType(Type type)
    {
        if (type == typeof(int) || type == typeof(int?))
            return "INT";
        
        if (type == typeof(long) || type == typeof(long?))
            return "BIGINT";
        
        if (type == typeof(string))
            return "NVARCHAR(MAX)";
        
        if (type == typeof(bool) || type == typeof(bool?))
            return "BIT";
        
        if (type == typeof(DateTime) || type == typeof(DateTime?))
            return "DATETIME";
        
        if (type == typeof(decimal) || type == typeof(decimal?))
            return "DECIMAL(18,2)";
        
        if (type == typeof(Guid) || type == typeof(Guid?))
            return "UNIQUEIDENTIFIER";
        
        return "NVARCHAR(MAX)";
    }
}

