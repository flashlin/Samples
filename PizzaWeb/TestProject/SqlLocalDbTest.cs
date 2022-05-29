using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using NUnit.Framework;
using PizzaWeb.Models;
using PizzaWeb.Models.Libs;
using T1.SqlLocalData;
using T1.Standard.IO;

namespace TestProject
{
	public class SqlLocalDbTest
	{
		private string _instanceName = "local_db_instance";
		private string _databaseName = "Northwind";
		private PizzaDbContext _db;
		private readonly SqlLocalDb _localDb = new SqlLocalDb(@"D:\Demo");

		[SetUp]
		public void Setup()
		{
			_localDb.EnsureInstanceCreated(_instanceName);
			_localDb.ForceDropDatabase(_instanceName, _databaseName);
			_localDb.DeleteDatabaseFile(_databaseName);
			_localDb.CreateDatabase(_instanceName, _databaseName);

			var factory = new SqlServerDbContextOptionsFactory(Options.Create(new PizzaDbConfig
			{
				ConnectionString  = "Server=(localdb)\\local_db_instance;Integrated security=SSPI;database=Northwind;"
			}));
			_db = new PizzaDbContext(factory.Create());
		}

		[Test]
		public void UpdateStoreShelvesById()
		{
			var sql = EmbeddedResource.GetEmbeddedResourceString(typeof(SqlLocalDbTest).Assembly, "PizzaDb.sql");
			_db.Database.ExecuteSqlRaw(sql);

			var items = _db.BannerTemplates.ToList();

			Assert.That(items.Count, Is.EqualTo(0));
		}
	}
}