using CommandLine;

namespace SqlSharp;

public class SqlSharpOptions
{
    public const string ExportTableData = "exportTableData";
    public const string ExtractCreateTableSql = "extractCreateTableSql";
    public const string ExtractSelectSql = "extractSelectSql";
    
    [Option('v', "Verb", Required = true, HelpText = "The action to perform. (export)")]
    public string ActionName { get; set; } = string.Empty;
    
    [Option('i', "Input", Required = false, HelpText = "Input file or folder")]
    public string Input { get; set; } = "";
    
    [Option('o', "Output", Required = false, HelpText = "Output file or folder")]
    public string Output { get; set; } = "";


    public bool IsActionName(string expectedActionName)
    {
        return ActionName.Equals(expectedActionName, StringComparison.InvariantCultureIgnoreCase);
    }
}