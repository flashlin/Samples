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




