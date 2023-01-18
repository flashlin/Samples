### We provide following features

* Simple Auto Mapping Clone Object Function
* Simple Auto Mapping any interface to WebApi Client Class


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

###  How to get Auto Mapping WebApiClient class ?
```csharp
[WebApiClient(ClientClassName = "SamApiClient", Namespace = "ConsoleDemoApp")]
//[AutoConstructorInject(typeof(IGlobalSetting), "globalSetting", "globalSetting")]
public interface IMyApiClient
{
   [WebApiClientMethod("mgmt/test1", Method = InvokeMethod.Get, Timeout = "00:00:10")]
   void Test(Request1 req);
    
   [WebApiClientMethod("mgmt/test2", Timeout = "00:00:30")]
   void Test2();

   [WebApiClientMethod("mgmt/test3")]
   Response1 Test3();
   
   [WebApiClientMethod("mgmt/test4")]
   Response1 Test4(int a);
}
```

Then in your application code, execute the code:
```csharp
using T1.SourceGenerator;

var client = new SamApiClient(null);
```

If you prefer a different namespace ("ConsoleDemoApp"), you can custom it.
```csharp
[WebApiClient(ClientClassName = "SamApiClient", Namespace = "ConsoleDemoApp")]
public interface IMyApiClient
{
    ...
}
```
