using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebSample.Services;

public class DelimitedStringJsonConverter : JsonConverter<string[]>
{
	public override string[]? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
	{
		return reader.GetString()?.Split(',');
	}

	public override void Write(Utf8JsonWriter writer, string[] value, JsonSerializerOptions options)
	{
		throw new NotImplementedException();
	}
}