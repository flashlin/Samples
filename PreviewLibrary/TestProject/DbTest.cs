using ExpectedObjects;
using PreviewLibrary.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace TestProject
{
	public class DbTest
	{
		[Fact]
		public async Task TestAsync()
		{
			var db = new MyDbContext();
			var resp = await db.QueryAsync<UserEntity>(new QueryCommand
			{
				Command = "select id,name,birth from [user]",
				Parameters = null
			});

			var data = resp.ToList();

			var expected = new List<UserEntity>()
			{
				new UserEntity() 
				{
					Id = 1,
					Name = "flash",
					Birth = DateTime.Parse("2022-03-01")
				}
			};

			expected.ToExpectedObject().ShouldEqual(data);
		}
	}
}
