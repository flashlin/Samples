using AspectCore.Extensions.DependencyInjection;
using FluentAssertions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using WebSample.Services;

namespace MyTests
{
	public class Tests
	{
		private IServiceCollection _services = null!;
		private IServiceProvider _sp = null!;
		private IUserService _service = null!;

		[SetUp]
		public void Setup()
		{
			_services = new ServiceCollection();
			_services.AddTransient<IUserService, UserService>();
			_services.AddTransient<IOptions<MemoryCacheOptions>>(sp => new MemoryCacheOptions());
			_services.AddSingleton<IMemoryCache, MemoryCache>();
			_sp = new DynamicProxyServiceProviderFactory().CreateServiceProvider(_services);
			_service = _sp.GetRequiredService<IUserService>();
		}
		
		[Test]
		public void Test1()
		{
			GivenCacheExists();

			var user = _service.GetUser("flash");
			
			user.Should().BeEquivalentTo(new UserInfo
			{
				Name = "flash",
				Seed = "123"
			});
		}

		private void GivenCacheExists()
		{
			var memoryCache = _sp.GetRequiredService<IMemoryCache>();
			var entry = memoryCache.CreateEntry("WebSample.Services.UserService.GetUser:flash");
			entry.AbsoluteExpiration = DateTimeOffset.Now.AddSeconds(10);
			entry.Value = new UserInfo
			{
				Name = "flash",
				Seed = "123"
			};
			memoryCache.Set(entry.Key, entry);
		}
	}
}