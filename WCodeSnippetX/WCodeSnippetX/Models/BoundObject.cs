using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using T1.Standard.Serialization;
using static System.Net.Mime.MediaTypeNames;

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

	public FormMainCef Form { get; set; } = null!;

	public void BringMeToFront()
	{
		Form.BringMeToFront();
	}

	public void Minimize()
	{
		Form.Minimize();
	}

	public string QueryCode(string text)
	{
		return _jsonSerializer.Serialize(_repo.QueryCode(text).ToList());
	}

	public void UpsertCode(string codeSnippetJson)
	{
		var codeSnippet = _jsonSerializer.Deserialize<CodeSnippetEntity>(codeSnippetJson);
		if (codeSnippet.Id == 0)
		{
			_repo.AddCode(codeSnippet);
			return;
		}
		_repo.UpdateCode(codeSnippet);
	}

	public void DeleteCode(int id)
	{
		_repo.DeleteCodeById(id);
	}

	public void SetClipboard(string text)
	{
		var t = new Thread(() =>
		{
			Clipboard.SetText(text);
		});
		t.SetApartmentState(ApartmentState.STA);
		t.Start();
	}
}

public static class UiThread
{
	public static void Execute(Action action)
	{
		var t = new Thread(() => action());
		t.SetApartmentState(ApartmentState.STA);
		t.Start();
	}
}