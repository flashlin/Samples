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
	public OkResult UnEcho(UnEchoRequest req)
	{
		_localQueryHostService.UnEcho(req);
		return Ok();
	}
}