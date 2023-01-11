

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
