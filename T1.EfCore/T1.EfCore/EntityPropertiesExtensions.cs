using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public static class EntityPropertiesExtensions
{
    public  static SqlRawProperty GetSqlRawProperty<T>(this IProperty property, int argumentIndex, T entity)
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

        var value = new ConstantValue{
            Value = rawValue, 
            Property = property,
            ArgumentIndex = argumentIndex,
        };
        var allowInsert = property.ValueGenerated == ValueGenerated.Never ||
                          property.GetAfterSaveBehavior() == PropertySaveBehavior.Save;
        return new SqlRawProperty
        {
            PropertyName = property.Name,
            ColumnName = columnName, 
            Value = value, 
            DefaultSql = defaultSql, 
            AllowInsert = allowInsert
        };
    }
}