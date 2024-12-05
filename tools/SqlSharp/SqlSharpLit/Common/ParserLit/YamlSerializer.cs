using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace SqlSharpLit.Common.ParserLit;

public class YamlSerializer
{
    private readonly IDeserializer _deserializer = new DeserializerBuilder()
        .WithNamingConvention(CamelCaseNamingConvention.Instance)
        .Build();

    public T Deserialize<T>(string yaml)
    {
        return _deserializer.Deserialize<T>(yaml);
    }
}