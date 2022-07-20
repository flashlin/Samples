namespace WebSample.Services;

public class GlobalSettingService : IGlobalSettingService
{
	private readonly IGlobalSettingRepo _globalSettingRepo;

	public GlobalSettingService(IGlobalSettingRepo globalSettingRepo)
	{
		_globalSettingRepo = globalSettingRepo;
	}

	public string GetStringValue(string key)
	{
		var setting = _globalSettingRepo
			.GetGlobalSettings()
			.FirstOrDefault(x => x.Id == key);
		if (setting == null)
		{
			return string.Empty;
		}
		return setting.Value;
	}

	public bool GetBoolValue(string key)
	{
		return string.Compare(GetStringValue(key), "true", StringComparison.OrdinalIgnoreCase) == 0;
	}
}