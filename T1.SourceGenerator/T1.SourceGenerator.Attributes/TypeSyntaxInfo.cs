namespace T1.SourceGenerator.Attributes;

public interface ISyntaxInfo
{
    
}

public class ArgumentSyntaxInfo
{
    public string Name { get; set; } = null!;
    public string TypeFullName { get; set; } = null!;
    public string ValueTypeFullName { get; set; } = null!;
    public object? Value { get; set; }

    public override string ToString()
    {
        return $"{TypeFullName} {Name}";
    }
}

public class AttributeSyntaxInfo
{
    public string TypeFullName { get; set; } = null!;
    public List<ArgumentSyntaxInfo> ConstructorArguments { get; set; } = null!;

    public ArgumentSyntaxInfo? GetArgumentSyntaxInfo(string name)
    {
        return ConstructorArguments
            .FirstOrDefault(x => x.Name == name);
    }

    public override string ToString()
    {
        var arguments = string.Join(",", ConstructorArguments.Select(x => x.ToString()));
        return $"{TypeFullName}({arguments})";
    }
}

public class ParameterSyntaxInfo
{
    public string TypeFullName { get; set; } = null!;
    public string Name { get; set; } = null!;
}

public class LambdaSyntaxInfo : ISyntaxInfo
{
    public List<ParameterSyntaxInfo> Parameters { get; set; } = new();
    public string Body { get; set; } = string.Empty;
}

public class MethodSyntaxInfo
{
    public string Name { get; set; } = null!;
    public List<ParameterSyntaxInfo> Parameters { get; set; } = new();
    public List<AttributeSyntaxInfo> Attributes { get; set; } = new();
    public string ReturnTypeFullName { get; set; } = null!;
    public string BodySourceCode { get; set; } = string.Empty;
}

public enum AccessibilityInfo
{
    Public,
    Private,
    Protected,
    Internal
}

public class PropertySyntaxInfo
{
    public AccessibilityInfo Accessibility { get; set; }
    public string TypeFullName { get; set; } = null!;
    public string Name { get; set; } = null!;
    public bool HasGetter { get; set; }
    public bool HasSetter { get; set; }
}


public class FieldSyntaxInfo
{
    public AccessibilityInfo Accessibility { get; set; }
    public string TypeFullName { get; set; } = null!;
    public string Name { get; set; } = null!;
    public bool IsReadOnly { get; set; }
    public List<AttributeSyntaxInfo> Attributes { get; set; } = new();
    public bool HasInitialization { get; set; }
    public ISyntaxInfo? InitializationCode { get; set; }
}

public class GenericArgumentSyntaxInfo
{
    public string TypeFullName { get; set; } = null!;
    public string Name { get; set; } = String.Empty;
}

public class FuncSyntaxInfo : ISyntaxInfo
{
    public List<string> GenericArguments { get; set; } = new();
    public string Body { get; set; } = string.Empty;
}

public class TypeSyntaxInfo
{
    public List<string> UsingNamespaces { get; set; } = new();
    public List<AttributeSyntaxInfo> Attributes { get; set; } = new();
    public string TypeFullName { get; set; } = null!;
    public List<MethodSyntaxInfo> Methods { get; set; } = new();
    public List<PropertySyntaxInfo> Properties { get; set; } = new();
    public List<string> BaseTypes { get; set; } = new();
    public List<FieldSyntaxInfo> Fields { get; set; } = new();

    public override string ToString()
    {
        return $"{TypeFullName}";
    }
}

public class EnumMemberSyntaxInfo
{
    public string Name { get; set; } = null!;
    public string Value { get; set; } = string.Empty;
}

public class EnumSyntaxInfo
{
    public string Name { get; set; } = null!;
    public List<EnumMemberSyntaxInfo> Values { get; set; } = new();
}
