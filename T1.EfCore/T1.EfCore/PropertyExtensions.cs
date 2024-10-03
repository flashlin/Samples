using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public static class PropertyExtensions
{
    public static SqlRawProperty GetSqlRawProperty<T>(this IProperty property, int argumentIndex, T entity)
    {
        var columnName = property.GetColumnName();
        var rawValue = property.PropertyInfo?.GetValue(entity);
        string? defaultSql = null;
        if (rawValue == null)
        {
            if (property.GetDefaultValue() != null)
                rawValue = property.GetDefaultValue();
            else
                defaultSql = property.GetDefaultValueSql();
        }

        var value = new ConstantValue
        {
            Property = property,
            Value = rawValue,
            ArgumentIndex = argumentIndex,
        };

        var allowInsert = property.ValueGenerated == ValueGenerated.Never ||
                          property.GetAfterSaveBehavior() == PropertySaveBehavior.Save;
        return new SqlRawProperty
        {
            Property = property,
            PropertyName = property.Name,
            ColumnName = columnName,
            DataValue = value,
            DefaultSql = defaultSql,
            AllowInsert = allowInsert
        };
    }
    
    public static bool IsAllowInsert(this IProperty property)
    {
        return property.ValueGenerated == ValueGenerated.Never ||
               property.GetAfterSaveBehavior() == PropertySaveBehavior.Save;
    }
}
