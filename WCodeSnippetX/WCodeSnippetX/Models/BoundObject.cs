using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;

namespace WCodeSnippetX.Models;

public class BoundObject : IBoundObject
{
	int _port;

	public BoundObject(IServer server)
	{
		var addressFeature = server.Features.Get<IServerAddressesFeature>()!;
		var address = addressFeature.Addresses.First();
		var idx = address.LastIndexOf(":");
		_port = int.Parse(address.Substring(idx+1));
	}

	public int GetPort()
	{
		return _port;
	}
}