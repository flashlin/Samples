using System.Reflection;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public class ConstantValue : IKnownValue
{
    public ConstantValue(object? value, IProperty? property = null, MemberInfo? memberInfo = null)
    {
        Value = value;
        Property = property;
        MemberInfo = memberInfo;
    }

    public object? Value { get; }
    public IProperty? Property { get; }
    public MemberInfo? MemberInfo { get; }
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