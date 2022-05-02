using SqlLocalDataTests.Repositories;
using System;
using System.IO;
using System.Linq;
using T1.SqlLocalData;
using T1.SqlLocalData.Extensions;
using Xunit;

namespace SqlLocalDataTests
{
	public class CreateTest : IClassFixture<InitializeFixture>
	{
		InitializeFixture _fixture;
		private readonly MyDbContext _myDb;

		public CreateTest(InitializeFixture fixture)
		{
			_fixture = fixture;
			_myDb = _fixture.GetMyDb();
			_fixture.CreateTable();
			_fixture.CreateSp();
		}

		[Fact]
		public void query_customer_by_my_db()
		{
			var customer = _myDb.Customers
				.First(x => x.Id == 3);

			Assert.Equal("Jack", customer.Name);
		}

		[Fact]
		public void execute_raw_sql()
		{
			
			var customer = _myDb.QuerySqlRaw<CustomerEntity>(@"select * from customer
where id = 3").First();

			Assert.Equal("Jack",customer.Name);
		}
		
		[Fact]
		public void execute_store_procedure()
		{
			var customer = _myDb.QuerySqlRaw<CustomerEntity>("EXEC MyGetCustomer @id", new
			{
				id = 3
			}).First();

			Assert.Equal("Jack",customer.Name);
		}
	}
}