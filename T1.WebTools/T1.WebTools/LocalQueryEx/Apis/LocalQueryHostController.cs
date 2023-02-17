using Microsoft.AspNetCore.Mvc;

namespace T1.WebTools.LocalQueryEx.Apis;

[ApiController]
[Route("api/[controller]/[action]")]
public class LocalQueryHostController : ControllerBase
{
	private readonly ILocalQueryHostService _localQueryHostService;

	public LocalQueryHostController(ILocalQueryHostService localQueryHostService)
	{
		_localQueryHostService = localQueryHostService;
	}
	
	[HttpPost]
	public EchoResponse Echo(EchoRequest req)
	{
		return _localQueryHostService.Echo(req);
	}

	[HttpPost]
	public void BindLocalQueryApp(BindLocalQueryAppRequest req)
	{
		_localQueryHostService.BindLocalQueryApp(req);
	}
}

public class BindLocalQueryAppRequest
{
	public string UniqueId { get; set; } = string.Empty;
	public string AppUid { get; set; } = string.Empty;
}