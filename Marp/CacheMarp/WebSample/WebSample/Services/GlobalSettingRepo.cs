namespace WebSample.Services;

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