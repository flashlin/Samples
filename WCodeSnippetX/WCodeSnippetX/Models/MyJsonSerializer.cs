using System.Text.Json;
using T1.Standard.Serialization;
using JsonSerializer = System.Text.Json.JsonSerializer;

namespace WCodeSnippetX.Models;

public class MyJsonSerializer : IJsonSerializer
{
	private readonly JsonSerializerOptions _options;

	public MyJsonSerializer()
	{
		_options = new JsonSerializerOptions()
		{
			PropertyNamingPolicy = JsonNamingPolicy.CamelCase
		};
	}

	public T Deserialize<T>(string json)
	{
		return JsonSerializer.Deserialize<T>(json, _options)!;
	}

	public object DeserializeObject(Type type, string json)
	{
		throw new NotImplementedException();
	}

	public object DeserializeTypedObject(string json)
	{
		throw new NotImplementedException();
	}

	public string Serialize<T>(T obj)
	{
		return JsonSerializer.Serialize(obj, _options);
	}

	public string SerializeWithType(object obj)
	{
		throw new NotImplementedException();
	}
}