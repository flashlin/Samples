using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using T1.Standard.Serialization;

namespace WCodeSnippetX.Models;

public class BoundObject : IBoundObject
{
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

	public string QueryCode(string text)
	{
		return _jsonSerializer.Serialize(_repo.QueryCode(text).ToList());
	}

	public FormMainCef Form { get; set; } = null!;

	public void Minimize()
	{
		Form.Minimize();
	}

	public void BringMeToFront()
	{
		Form.BringMeToFront();
	}
}