using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public static class EntityPropertiesExtensions
{
    public  static SqlRawProperty GetSqlRawProperty<T>(this IProperty p, int argumentIndex, T entity)
    {
        var columnName = p.GetColumnName();
        var rawValue = p.PropertyInfo?.GetValue(entity);
        string? defaultSql = null;
        if (rawValue == null)
        {
            if (p.GetDefaultValue() != null)
                rawValue = p.GetDefaultValue();
            else
                defaultSql = p.GetDefaultValueSql();
        }

        var value = new ConstantValue{
            Value = rawValue, 
            Property = p,
            ArgumentIndex = argumentIndex,
        };
        var allowInsert = p.ValueGenerated == ValueGenerated.Never ||
                          p.GetAfterSaveBehavior() == PropertySaveBehavior.Save;
        return new SqlRawProperty
        {
            PropertyName = p.Name,
            ColumnName = columnName, 
            Value = value, 
            DefaultSql = defaultSql, 
            AllowInsert = allowInsert
        };
    }
}