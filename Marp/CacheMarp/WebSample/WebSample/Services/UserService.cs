namespace WebSample.Services
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
}
