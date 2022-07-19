using System.Text;
using System.Text.Json.Serialization;
using AspectCore.DynamicProxy;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Caching.Memory;
using T1.Standard.Serialization;

namespace WebSample.Services;

public class CacheInterceptorAttribute : AbstractInterceptorAttribute
{
	private static readonly SemaphoreSlim _locker = new(1, 1);
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

	private static string CreateCacheKey(AspectContext context)
	{
		var sb = new StringBuilder();
		foreach (var item in context.ServiceMethod.GetParameters())
		{
			sb.Append(item.Name);
		}
		foreach (var value in context.Parameters)
		{
			sb.Append(value.GetHashCode());
		}
		return $"{context.ImplementationMethod.DeclaringType}.{context.ImplementationMethod.Name}." + sb.ToString();
	}
}






public class DistributedCacheInterceptorAttribute : AbstractInterceptorAttribute
{
	private readonly SemaphoreSlim _locker = new SemaphoreSlim(1, 1);
	private readonly TimeSpan _timeout = TimeSpan.FromSeconds(1);

	public override async Task Invoke(AspectContext context, AspectDelegate next)
	{
		var cacheKey = CreateCacheKey(context);
		var cache = context.ServiceProvider.GetService<IDistributedCache>()!;
		var jsonSerializer = context.ServiceProvider.GetService<IJsonSerializer>()!;
		await _locker.WaitAsync();
		try
		{
			var result = await cache.GetStringAsync(cacheKey).WaitAsync(_timeout);
			if (!string.IsNullOrEmpty(result))
			{
				context.ReturnValue = result;
				return;
			}
			await next(context);
			if (context.ReturnValue != null)
			{
				var json = jsonSerializer.Serialize(context.ReturnValue);
				await cache.SetStringAsync(cacheKey, json).WaitAsync(_timeout);
			}
		}
		finally
		{
			_locker.Release();
		}
	}

	private static string CreateCacheKey(AspectContext context)
	{
		var sb = new StringBuilder();
		foreach (var item in context.ServiceMethod.GetParameters())
		{
			sb.Append(item.Name);
		}
		foreach (var value in context.Parameters)
		{
			sb.Append(value.GetHashCode());
		}
		return $"{context.ImplementationMethod.DeclaringType}.{context.ImplementationMethod.Name}." + sb.ToString();
	}
}