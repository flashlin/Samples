using System.Text;

namespace SqlSharpLit.Common.ParserLit;

public static class StreamWriterCreator
{
    public static StreamWriter Create(string createTablesFile)
    {
        var fileStream = new FileStream(createTablesFile, FileMode.Create);
        var writer = new StreamWriter(fileStream, Encoding.UTF8);
        return writer;
    }
}