using CommandLine;

namespace SqlSharp;

public class SqlSharpOptions
{
    public const string ExportTableData = "exportTableData";
    public const string ExtractCreateTableSql = "extractCreateTableSql";
    public const string ExtractSelectSql = "extractSelectSql";
    public const string GenerateDatabaseDescriptionsMdFile = "generateDatabaseDescriptionsMdFile";
    
    [Option('v', "Verb", Required = true, HelpText = "The action to perform. (export)")]
    public string ActionName { get; set; } = string.Empty;
    
    [Option('i', "Input", Required = false, HelpText = "Input file or folder")]
    public string Input { get; set; } = "";
    
    [Option('o', "Output", Required = false, HelpText = "Output file or folder")]
    public string Output { get; set; } = "";

    [Option( "DatabaseNamePathDeep", Required = false, HelpText = "DatabaseNamePath deep")]
    public int DatabaseNamePathDeep { get; set; } = 6;

    public bool IsActionName(string expectedActionName)
    {
        return ActionName.Equals(expectedActionName, StringComparison.InvariantCultureIgnoreCase);
    }
}