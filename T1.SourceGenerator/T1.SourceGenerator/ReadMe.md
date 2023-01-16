### How do I get started ?

First, install T1.SourceGenerator nuget package.
Attach AutoMapping Attribute on any class.

```csharp
using T1.SourceGenerator.Attributes;
namespace MyDemoApp;

[AutoMapping(typeof(Employee))]
public class User
{
    public string Name { get; set; }
    public int Level { get; set; }
    public int Vip { get; set; }
}

public class Employee 
{
    public string Name { get; set; }
    public float Level { get; set; }
    public int Vip { get; }
}
```

Then in your application code, execute the mappings:
```csharp
var source = new User();
var dto = source.ToEmployee(); 
```

If you hope change `ToEmployee()` method to other extension method, you can do this:
```csharp
[AutoMapping(typeof(Employee), "CloneToEmployee")]
public class User
{
    public string Name { get; set; }
    public int Level { get; set; }
    public int Vip { get; set; }
}
```

Then in your application code, execute the mappings:
```csharp
var source = new User();
var dto = source.CloneToEmployee(); 
```
