using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebSample.Services;

public class BooleanStringJsonConverter : JsonConverter<bool>
{
	public override bool Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
	{
		return reader.GetString() == "true";
	}

	public override void Write(Utf8JsonWriter writer, bool value, JsonSerializerOptions options)
	{
		throw new NotImplementedException();
	}
}