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

	public List<UnbindLocalQueryAppInfo> GetUnbindLocalQueryApps()
	{
		return _localQueryHostService.GetUnbindLocalQueryApps()
			.Select(x => new UnbindLocalQueryAppInfo
			{
				AppUid = x.AppUid,
				Port = x.Port,
			})
			.ToList();
	}

	[HttpPost]
	public EchoResponse Echo(EchoRequest req)
	{
		return _localQueryHostService.Echo(req);
	}

	[HttpPost]
	public BindLocalQueryAppResponse BindLocalQueryApp(BindLocalQueryAppRequest req)
	{ 
        return _localQueryHostService.BindLocalQueryApp(req);
	}
}

public class UnbindLocalQueryAppInfo
{
	public string AppUid { get; set; } = string.Empty;
	public int Port { get; set; }
}

public class BindLocalQueryAppRequest
{
	public string UniqueId { get; set; } = string.Empty;
	public string AppUid { get; set; } = string.Empty;
}