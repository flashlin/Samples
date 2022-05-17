using System.Reflection;

namespace PizzaWeb.Models.Runtime;

public class CSharpCodeExecutor
{
    private readonly Assembly _assembly;

    public CSharpCodeExecutor(byte[] assemblyBytes)
    {
        _assembly = Assembly.Load(assemblyBytes);
    }

    public object? Execute(string typeFullname, string methodName, object[] arguments)
    {
        var type = _assembly.GetType(typeFullname)!;
        var obj = Activator.CreateInstance(type);
        return type.InvokeMember(methodName,
            BindingFlags.Default | BindingFlags.InvokeMethod,
            null,
            obj,
            arguments);
    }
}