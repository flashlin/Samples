using SqlLocalDataTests.Repositories;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.EntityFrameworkCore;
using T1.SqlLocalData;
using T1.SqlLocalData.Extensions;
using Xunit;

namespace SqlLocalDataTests
{
	//[TestCaseOrderer("SqlLocalDataTests.AllowReversePInvokeCallsAttribute", "SqlLocalDataTests")]
	public class FastCreateTest : IClassFixture<InitializeFixture>
	{
		private MyDbContext _myDb;
		private InitializeFixture _fixture;

		public FastCreateTest(InitializeFixture fixture)
		{
			_fixture = fixture;
			_myDb = fixture.GetMyDb();
		}

		[Fact]
		public void execute_raw_sql()
		{
			DropTable();
			_fixture.CreateTable();

			var customer = _myDb.Customers
				 .First(x => x.Id == 3);

			Assert.Equal("Jack", customer.Name);
		}

		[Fact]
		public void execute_raw_sql_from_file()
		{
			DropTable();
			DropSp("MyGetCustomer");
			_myDb.ExecuteSqlRawFromFile("./Contents/CreateTable.sql");
			_myDb.ExecuteSqlRawFromFile("./Contents/MyGetCustomer.sql");

			var customer = _myDb.Customers
				 .First(x => x.Id == 3);

			Assert.Equal("Jack", customer.Name);
		}

		[Fact]
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

			Assert.Equal("Jack", customer.Name);
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