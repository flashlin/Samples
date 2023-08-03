using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using T1.SourceGenerator.ApiClientGen;
using T1.SourceGenerator.Attributes;
using T1.SourceGenerator.Utils;

namespace T1.SourceGenerator.LinqExpressionGen;

[Generator]
public class LinqExpressionGenerator : ISourceGenerator
{
    private readonly EmbeddedSources _embedded = new EmbeddedSources(typeof(LinqExpressionGenerator).Assembly);

    public void Initialize(GeneratorInitializationContext context)
    {
        context.RegisterForPostInitialization(i =>
        {
            i.AddSource("LinqExpressionCompileAttribute.g.cs", 
                _embedded.LoadTemplateForEmitting("LinqExpressionCompileAttribute"));
        });
    }

    public void Execute(GeneratorExecutionContext context)
    {
        var compilation = context.Compilation;
        QueryLinqExpression(compilation);
        
        foreach (var type in compilation.GetAllTypes())
        {
            var classFullName = type.TypeFullName;
            //var attrs = compilation.GetAttributeDeclarations<LinqExpressionCompileAttribute>(type);
            foreach (var field in type.Fields)
            {
                var linqExprCompileAttr = field.Attributes
                    .FirstOrDefault(x => x.TypeFullName == typeof(LinqExpressionCompileAttribute).FullName);
                if (linqExprCompileAttr == null)
                {
                    continue;
                }
                Console.WriteLine(field.InitializationCode);
            }
        }


        var syntaxTrees = compilation.SyntaxTrees;
        foreach (var syntaxTree in syntaxTrees)
        {
            var root = syntaxTree.GetRoot();
            var semanticModel = compilation.GetSemanticModel(syntaxTree);

            // 找到类声明语法节点
            var classDeclaration = root.DescendantNodes().OfType<ClassDeclarationSyntax>()
                .FirstOrDefault(c => c.Identifier.ValueText == "MyDbContext");

            // 找到方法声明语法节点
            var methodDeclaration = classDeclaration?.DescendantNodes().OfType<MethodDeclarationSyntax>()
                .FirstOrDefault(m => m.Identifier.ValueText == "GetUserByIdInternal");

            if (methodDeclaration != null)
            {
                var code = methodDeclaration.ToString();
                Console.WriteLine("GetUserByIdInternal code:\n" + code);
            }
        }
    }

    private void QueryLinqExpression(Compilation compilation)
    {
        var syntaxTrees = compilation.SyntaxTrees;

        // 遍历每个语法树
        foreach (var syntaxTree in syntaxTrees)
        {
            // 查找查询表达式的节点
            var queryExpression = syntaxTree.GetRoot()
                .DescendantNodes()
                .OfType<Microsoft.CodeAnalysis.CSharp.Syntax.QueryExpressionSyntax>()
                .FirstOrDefault();

            if (queryExpression != null)
            {
                // 将查询表达式转换为字符串
                var expressionString = queryExpression.ToString();
                
                Console.WriteLine("");

                // 将提取的表达式添加到生成的代码中
                // var sourceText = SourceText.From(expressionString, System.Text.Encoding.UTF8);
                // var hintName = "ExtractedExpression.g.cs";
                // context.AddSource(hintName, sourceText);
            }
        }
    }
}