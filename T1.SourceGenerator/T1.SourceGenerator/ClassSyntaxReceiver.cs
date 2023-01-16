using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using T1.SourceGenerator.AutoMappingSlim;

namespace T1.SourceGenerator;

public class ClassSyntaxReceiver : ISyntaxReceiver
{
    public List<AutoMappingMetadata> AutoMappingMetadataList { get; set; } = new();

    public void OnVisitSyntaxNode(SyntaxNode syntaxNode)
    {
        if (syntaxNode is not ClassDeclarationSyntax cds)
        {
            return;
        }

        var attributeSyntax = cds.AttributeLists
            .SelectMany(x => x.Attributes)
            .FirstOrDefault(attr => attr.Name.ToString() == "AutoMapping");
        if (attributeSyntax == null)
        {
            return;
        }

        var attributeArguments = attributeSyntax.ArgumentList!.Arguments;
        var fromTypeClassName = attributeArguments.First().GetTypeofClassName();
        var toTypeClassName = attributeArguments.ElementAt(1).GetTypeofClassName();
        var className = cds.Identifier.Text;

        // cds.Members.Select(x => x as PropertyDeclarationSyntax)
        //     .Where(x => x != null)
        //     .Select(x => x!.Identifier.Text)
        //     .ToList();
        AutoMappingMetadataList.Add(new AutoMappingMetadata
        {
            FromTypeName = fromTypeClassName,
            ToTypeName = toTypeClassName,
            ClassName = className,
        });
    }
}