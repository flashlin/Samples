namespace WebSample.Services;

public interface IGlobalSettingService
{
	string GetStringValue(string key);
	bool GetBoolValue(string key);
}

public interface IGlobalSettingFactory<out T> 
	where T : new()
{
	T Create();
}

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

public interface IGlobalSettingRepo
{
	List<GlobalSetting> GetGlobalSettings();
}

public class GlobalSettingRepo : IGlobalSettingRepo
{
	public List<GlobalSetting> GetGlobalSettings()
	{
		return new List<GlobalSetting>()
		{
			new()
			{
				Id = "Countries",
				Value = "CN,TH,US"
			},
			new()
			{
				Id = "FeatureEnabled",
				Value = "true"
			},
			new()
			{
				Id = "CustomerIds",
				Value = "[1,2,3]"
			},
			new()
			{
				Id = "BlockedDomains",
				Value = "[{\"Country\":\"CN\"},{\"Country\":\"TW\"}]",
			}
		};
	}
}

public class BlockedDomain
{
	public string Country { get; set; }
}

public interface IAbstractFactory<out T>
{
	T Create();
}

public class AbstractFactory<T> : IAbstractFactory<T>
{
	private readonly Func<T> _factoryFn;

	public AbstractFactory(Func<T> factoryFn)
	{
		_factoryFn = factoryFn;
	}

	public T Create()
	{
		return _factoryFn();
	}
}