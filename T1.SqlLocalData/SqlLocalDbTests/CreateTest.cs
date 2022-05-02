using NUnit.Framework;
using SqlLocalDataTests.Repositories;
using System;
using System.IO;
using System.Linq;
using T1.SqlLocalData;
using T1.SqlLocalData.Extensions;

namespace SqlLocalDataTests
{
	public class CreateTest
	{
		InitializeFixture _fixture;
		private MyDbContext _myDb;

		[SetUp]
		public void Initialize()
		{
			_fixture = new InitializeFixture();
			_myDb = _fixture.GetMyDb();
			_fixture.CreateTable();
			_fixture.CreateSp();
		}

		[Test]
		public void query_customer_by_my_db()
		{
			var customer = _myDb.Customers
				.First(x => x.Id == 3);

			Assert.AreEqual("Jack", customer.Name);
		}

		[Test]
		public void execute_raw_sql()
		{
			
			var customer = _myDb.QuerySqlRaw<CustomerEntity>(@"select * from customer
where id = 3").First();

			Assert.AreEqual("Jack",customer.Name);
		}
		
		[Test]
		public void execute_store_procedure()
		{
			var customer = _myDb.QuerySqlRaw<CustomerEntity>("EXEC MyGetCustomer @id", new
			{
				id = 3
			}).First();

			Assert.AreEqual("Jack",customer.Name);
		}
	}
}