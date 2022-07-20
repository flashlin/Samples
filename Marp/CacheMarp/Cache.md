--- 
marp: true
theme: gaia
paginate: true
---

以下程式碼混雜, 掩蓋商業邏輯核心, 不易閱讀
```C#
public class UserService
{
   Dictionary<string, UserInfo> _cache = new();
   public UserInfo GetUser(string name)
   {
      if(_cache.TryGetValue(name, out var oldUserInfo))
      {
         return oldUserInfo;
      }
      ... = _userRepo.QueryByName(name);
      ... some business logical ...
      var newUserInfo = ...
      _cache[name] = newUserInfo;
      return newUserInfo;
   }
}
```

---

單一功能原則
```C#
public class UserService : IUserService
{
   public UserInfo GetUser(string name)
   {
      ... = _userRepo.QueryByName(name);
      ... some business logical ...
      return newUserInfo;
   }
}
```

---

將物件有效的往上附加職責, 不動到內部的程式碼, 在原來職責上附加額外的職責
```C#
public class CachedUserService : IUserService
{
   IDictionary<string, UserInfo> _cache = new ConcurrentDictionary<string, UserInfo>();
   ...
   public UserInfo GetUser(string name)
   {
      if(_cache.TryGetValue(name, out var oldUserInfo))
      {
         return oldUserInfo;
      }
      _cache[name] = _origin.GetUser();
      return newUserInfo;
   }
}
```


---

以上是裝飾者模式(Decorator Pattern) 很棒, 但可以更好
針對重複發生的需求
```C#
public interface IUserService
{
   [CacheInterceptor]
   UserInfo GetUser(string name);
}
```


---

```
public class CacheInterceptorAttribute : AbstractInterceptorAttribute
{
	private readonly SemaphoreSlim _locker = new(1, 1);
	public override async Task Invoke(AspectContext context, AspectDelegate next)
	{
		var cacheKey = CreateCacheKey(context);
		var cache = context.ServiceProvider.GetService<IMemoryCache>();
		await _locker.WaitAsync();
		try
		{
			var result = await cache.GetOrCreateAsync(cacheKey, async (_) =>
			{
				await next(context);
				return context.ReturnValue;
			});
			context.ReturnValue = result;
		}
		finally
		{
			_locker.Release();
		}
	}
}
```


---


* 單一功能原則
* 針對重複發生的功能邏輯, 用不同的面向去思考
* 可避免後人重複犯以前同樣的錯誤 (例如 lock, 避免同一時間大量 user 存取同一個 resource)


---


```C#
public IActionResult Index()
{
	var featureEnabled1 = _globalSettingService.GetStringValue("FeatureEnabled") == "true";
	var featureEnabled2 = _globalSettingService.GetBoolValue("FeatureEnabled");
   ...
	return View;
}
```

---

Create strongly typed config object
```C#
public class MyGlobalSettings
{
   public string[] Countries { get; set; } = Array.Empty<string>();
   public bool FeatureEnabled { get; set; } = false;
   public List<int> CustomerIds { get; set; } = new();
   public List<BlockedDomain> BlockedDomains { get; set; } = new();
}
```

---

Use Factory Pattern Create our GlobalSetting Config Value
```C#
public HomeController(IGlobalSettingFactory<MyGlobalSettings> globalSettingFactory)
{
	_myGlobalSettings = globalSettingFactory.Create();
}

public IActionResult Index()
{
	//var featureEnabled1 = _globalSettingService.GetStringValue("FeatureEnabled") == "true";
	//var featureEnabled2 = _globalSettingService.GetBoolValue("FeatureEnabled");
	var featureEnabled3 = _myGlobalSettings.FeatureEnabled;
   ...
}
```

---

```C#
public T Create()
{
   var json = new StringBuilder();
   json.AppendLine("{");
   var allSettings = _globalSettingRepo.GetGlobalSettings().ToArray();
   foreach (var setting in allSettings)
   {
      json.Append($"\"{setting.Id}\" : ");
   	json.Append(IsSurroundWithJson(setting) ? $"{setting.Value}" : $"\"{setting.Value}\"");
   	if (setting != allSettings.Last())
   	{
   	   json.Append(",");
   	}
   }
   json.AppendLine("}");
   return Deserialize(json);
}
```


---

Register
```C#

Services.AddTransient<IGlobalSettingFactory<MyGlobalSettings>, GlobalSettingFactory<MyGlobalSettings>>();

```