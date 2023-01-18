namespace T1.SourceGenerator.Attributes;

[AttributeUsage(AttributeTargets.Interface, Inherited = false, AllowMultiple = true)]
public class WebApiClientConstructorInjectAttribute : Attribute
{
    public WebApiClientConstructorInjectAttribute(Type interfaceType, string parameterName)
    {
        InterfaceType = interfaceType;
        ParameterName = parameterName;
    }
    public Type InterfaceType { get; set; }
    public string ParameterName { get; set; }
    public string AssignCode { get; set; } = string.Empty;
}