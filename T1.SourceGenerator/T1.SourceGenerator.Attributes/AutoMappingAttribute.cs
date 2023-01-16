namespace T1.SourceGenerator.Attributes;

[AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
//[System.Diagnostics.Conditional("STRONGLY_TYPED_ID_USAGES")]
public class AutoMappingAttribute : Attribute
{
    public AutoMappingAttribute(Type toType, string? mappingMethodName = null)
    {
        this.ToType = toType;
        MappingMethodName = mappingMethodName ?? toType.Name;
    }

    public Type ToType { get; set; }
    public string MappingMethodName { get; set; }
}