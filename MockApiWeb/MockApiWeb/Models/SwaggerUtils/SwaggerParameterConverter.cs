using System.Text.Json;
using System.Text.Json.Serialization;

namespace MockApiWeb.Models.SwaggerUtils;

public class SwaggerParameterConverter : JsonConverter<SwaggerParameter>
{
    public override bool CanConvert(Type typeToConvert)
    {
        return true;
    }

    public override SwaggerParameter? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        throw new NotImplementedException();
    }

    public override void Write(Utf8JsonWriter writer, SwaggerParameter value, JsonSerializerOptions options)
    {
        throw new NotImplementedException();
    }
}