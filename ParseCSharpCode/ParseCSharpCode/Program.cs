using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

var code = @"
(IEmployee custom, IOptions<AppConfig> config, ILogger logger)";


var tree = CSharpSyntaxTree.ParseText(code);
var root = tree.GetCompilationUnitRoot();

Console.WriteLine($"The tree is a {root.Kind()} node.");
Console.WriteLine($"The tree has {root.Members.Count} elements in it.");
Console.WriteLine($"The tree has {root.Usings.Count} using statements. They are:");
foreach (UsingDirectiveSyntax element in root.Usings)
	Console.WriteLine($"\t{element.Name}");


var member = root.Members.First();
Console.WriteLine($"{member.Kind()}");
var globalStatement = (GlobalStatementSyntax)member;

var s = globalStatement.Statement;

Console.ReadLine();
