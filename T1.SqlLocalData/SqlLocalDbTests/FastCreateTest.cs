using System.Linq;
using Microsoft.EntityFrameworkCore;
using NUnit.Framework;
using SqlLocalDbTests.Repositories;

namespace SqlLocalDbTests
{
	public class FastCreateTest
	{
		private MyDbContext _myDb;
		private InitializeFixture _fixture;

		[SetUp]
		public void Initialize()
		{
			_fixture = new InitializeFixture();
			_myDb = _fixture.GetMyDb();
		}

		[Test]
		public void execute_raw_sql()
		{
			DropTable();
			_fixture.CreateTable();

			var customer = _myDb.Customers
				 .First(x => x.Id == 3);

			Assert.AreEqual("Jack", customer.Name);
		}

		[Test]
		public void execute_raw_sql_from_file()
		{
			DropTable();
			DropSp("MyGetCustomer");
			_myDb.ExecuteSqlRawFromFile("./Contents/CreateTable.sql");
			_myDb.ExecuteSqlRawFromFile("./Contents/MyGetCustomer.sql");

			var customer = _myDb.Customers
				 .First(x => x.Id == 3);

			Assert.AreEqual("Jack", customer.Name);
		}

		[Test]
		public void execute_store_procedure()
		{
			DropTable();
			DropSp("MyGetCustomer");
			_fixture.CreateTable();
			_fixture.CreateSp();

			var customer = _myDb.QuerySqlRaw<CustomerEntity>("EXEC MyGetCustomer @id", new
			{
				id = 3
			}).First();

			Assert.AreEqual("Jack", customer.Name);
		}

		private void DropTable()
		{
			_myDb.Database.ExecuteSqlRaw($@"DROP TABLE IF EXISTS Customer");
		}

		private void DropSp(string spName)
		{
			_myDb.Database.ExecuteSqlRaw($@"DROP PROCEDURE IF EXISTS {spName}");
		}
	}
}