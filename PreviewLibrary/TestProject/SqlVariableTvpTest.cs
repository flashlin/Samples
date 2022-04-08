using System;
using Xunit;
using Microsoft.SqlServer.Server;
using System.Collections.Generic;
using System.Linq;
using PreviewLibrary.RecursiveParser;

namespace TestProject
{
	public class SqlVariableTvpTest
	{
		public class User
		{
			public int Id { get; set; }
			public string Name { get; set; }
			public DateTime Birth { get; set; }
			public decimal Price { get; set; }
			public bool IsUat { get; set; }
		}

		[Fact]
		public void Test1()
		{
			var user = new User()
			{
				Id = 1,
				Name = "flash",
				Birth = DateTime.Parse("2022-03-04"),
				Price = 123.1m,
				IsUat = true
			};

			var tvp = user.ToSqlVariableTvp();

			var tvpDump = string.Join("\r\n", tvp.Select(x => $"{x.Dump()}"));


			var expected = @"{Name:'@Id',DataType:'int',DataValue:1}
{Name:'@Name',DataType:'nvarchar(50)',DataValue:flash}
{Name:'@Birth',DataType:'datetime',DataValue:3/4/2022 12:00:00 AM}
{Name:'@Price',DataType:'decimal(18,3)',DataValue:123.1}
{Name:'@IsUat',DataType:'bit',DataValue:True}";
			Assert.Equal(expected, tvpDump);
		}
	}
}