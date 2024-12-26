using System.Text.Json;

namespace SqlSharpLit.Common.ParserLit;

public class JsonDocSerializer
{
    public string Serialize<T>(T obj)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };
        return JsonSerializer.Serialize(obj, options);
    }

    public void WriteToJsonFile<T>(T obj, string jsonFile)
    {
        using var writer = StreamWriterCreator.Create(jsonFile);
        var json = Serialize(obj);
        writer.Write(json);
        writer.Flush();
    }
}