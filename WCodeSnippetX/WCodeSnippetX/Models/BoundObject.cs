using System.Text.Json;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using T1.Standard.Serialization;
using JsonSerializer = System.Text.Json.JsonSerializer;

namespace WCodeSnippetX.Models;

public class BoundObject : IBoundObject
{
	//private readonly IServiceProvider _serviceProvider;
	private readonly IJsonSerializer _jsonSerializer;
	private readonly ICodeSnippetRepo _repo;
	//readonly int _port;

	//public BoundObject(IServer server, IServiceProvider serviceProvider,
	//	IJsonSerializer jsonSerializer)
	//{
	//	_serviceProvider = serviceProvider.CreateScope().ServiceProvider;
	//	_jsonSerializer = jsonSerializer;
	//	var addressFeature = server.Features.Get<IServerAddressesFeature>()!;
	//	var address = addressFeature.Addresses.First();
	//	var idx = address.LastIndexOf(":");
	//	_port = int.Parse(address.Substring(idx+1));
	//}

	public BoundObject(ICodeSnippetRepo repo, IJsonSerializer jsonSerializer)
	{
		_repo = repo;
		_jsonSerializer = jsonSerializer;
	}

	//public int GetPort()
	//{
	//	return _port;
	//}

	public string QueryCodeAsync(string text)
	{
		return _jsonSerializer.Serialize(_repo.QueryCode(text).ToList());
	}
}

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