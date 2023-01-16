using Microsoft.CodeAnalysis;

namespace T1.SourceGenerator.Utils;

public static class GeneratorExecutionContextExtension
{
    public static FileContentInfo? GetProjectFileContent(this GeneratorExecutionContext context, string filename)
    {
        var file = context.AdditionalFiles.FirstOrDefault(e => e.Path.EndsWith(filename));
        if (file == null)
        {
            return null;
        }
        var fileDirectory = Path.GetDirectoryName(file.Path)!;
        var fileContent = File.ReadAllText(file.Path);
        return new FileContentInfo
        {
            Directory = fileDirectory,
            Content = fileContent,
        };
    }
}