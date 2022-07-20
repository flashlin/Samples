using System.Runtime.Serialization;
using System.Text.RegularExpressions;
using LanguageExt.Common;

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

	public class UserTypingValidator
	{
		public void CheckName(string name)
		{
			if (string.IsNullOrEmpty(name))
			{
				throw new InvalidDataContractException("Content cannot be blank");
			}
			if (name.Any(x => "~`!@#$%^&*()-_+={}[]|\\:\";'<>,.?/".Contains(x)))
			{
				throw new InvalidDataContractException("Content cannot contain special characters");
			}
			if (name.Length < 3)
			{
				throw new InvalidDataContractException("Content length must be at least 3");
			}
		}
	}

	public class UserTypingValidator2
	{
		public Result<bool> CheckName(string name)
		{
			if (string.IsNullOrEmpty(name))
			{
				return new(new InvalidDataContractException("Content cannot be blank"));
			}

			if (name.Any(x => "~`!@#$%^&*()-_+={}[]|\\:\";'<>,.?/".Contains(x)))
			{
				return new(new InvalidDataContractException("Content cannot contain special characters"));
			}
			if (name.Length < 3)
			{
				return new(new InvalidDataContractException("Content length must be at least 3"));
			}
			return true;
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

		public bool Check2(string name)
		{
			var rc = new UserTypingValidator2().CheckName(name);
			return rc.IsSuccess;
		}
	}

	public class GlobalSetting
	{
		public string Id { get; set; }

		public string Value { get; set; }

		public string Description { get; set; }

		public bool IsUat { get; set; }
	}
}
