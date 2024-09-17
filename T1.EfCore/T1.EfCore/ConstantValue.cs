using System.Reflection;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public class ConstantValue : IKnownValue
{
    public object? Value { get; set; }
    public IProperty? Property { get; set; }
    public MemberInfo? MemberInfo { get; set; }
    public int ArgumentIndex { get; set; }

    public IEnumerable<ConstantValue> GetConstantValues()
    {
        yield return this;
    }

    public IEnumerable<PropertyValue> GetPropertyValues()
    {
        return Array.Empty<PropertyValue>();
    }
}