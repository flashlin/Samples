using System.ComponentModel;
using System.Globalization;
using System.Text.Json;

namespace QueryWeb.Models;

//[TypeConverter(typeof(CustomItemTypeConverter))]
public class SelectItem
{
    public string Text { get; set; }
    public object Value { get; set; }
}

public class CustomItemTypeConverter : TypeConverter
{
    public override bool CanConvertFrom(ITypeDescriptorContext? context, Type sourceType)
    {
        return sourceType == typeof(string);
    }

    public override object? ConvertFrom(ITypeDescriptorContext? context, CultureInfo? culture, object value)
    {
        if (value is string stringValue)
        {
            if (!string.IsNullOrEmpty(stringValue))
            {
                return JsonSerializer.Deserialize<SelectItem>(stringValue)!;
            }
        }
        return base.ConvertFrom(context, culture, value);
    }

    public override bool CanConvertTo(ITypeDescriptorContext? context, Type? destinationType)
    {
        return destinationType == typeof(string);
    }

    public override object? ConvertTo(ITypeDescriptorContext? context, CultureInfo? culture, object? value, Type destinationType)
    {
        if (destinationType == typeof(string))
        {
            if (value == null)
            {
                return string.Empty;
            }
            return JsonSerializer.Serialize(value);
        }
        return base.ConvertTo(context, culture, value, destinationType);
    }
}