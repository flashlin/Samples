using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AspectCore.DynamicProxy;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;
using System.Linq;

namespace FrameworkWebSite.Services
{
	public interface IUserService
	{
		[CacheInterceptor]
		UserInfo GetUser(string name);
	}

	public class UserService : IUserService
	{
		public UserInfo GetUser(string name)
		{
			Thread.Sleep(3000);
			return new UserInfo
			{
				Name = name,
			};
		}
	}

	public class UserInfo
	{
		public string Name { get; set; } = string.Empty;
		public string Seed { get; set; } = $"{new Random().Next(100) + 1}";
	}

	public class UserTypingValidator
	{
		public void CheckName(string name)
		{
			if (string.IsNullOrEmpty(name))
			{
				throw new Exception("Content cannot be blank");
			}
			if (name.Any(x => "~`!@#$%^&*()-_+={}[]|\\:\";'<>,.?/".Contains(x)))
			{
				throw new Exception("Content cannot contain special characters");
			}
			if (name.Length < 3)
			{
				throw new Exception("Content length must be at least 3");
			}
		}
	}

	public class MyService
	{
		public bool Check(string name)
		{
			try
			{
				new UserTypingValidator().CheckName(name);
				return true;
			}
			catch
			{
				return false;
			}
		}
	}

	public class GlobalSetting
	{
		public string Id { get; set; }

		public string Value { get; set; }

		public string Description { get; set; }

		public bool IsUat { get; set; }
	}
	
	public class CacheInterceptorAttribute : AbstractInterceptorAttribute
	{
		private static readonly SemaphoreSlim _locker = new SemaphoreSlim(1, 1);
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

}
