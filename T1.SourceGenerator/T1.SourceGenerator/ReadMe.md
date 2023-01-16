

```
<ItemGroup>
  <ProjectReference Include="..\T1.SourceGenerator\T1.SourceGenerator.csproj"
                    OutputItemType="Analyzer" ReferenceOutputAssembly="false" />
</ItemGroup>
```

```
dotnet add package Microsoft.CodeAnalysis.CSharp
dotnet add package Microsoft.CodeAnalysis.Analyzers
```


```csharp
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