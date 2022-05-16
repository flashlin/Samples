using Microsoft.EntityFrameworkCore;
using NUnit.Framework;
using PizzaWeb.Models;
using System.Text.Json;
using System.Text.Json.Nodes;
using PizzaWeb.Models.Libs;
using T1.Standard.IO;

namespace TestProject
{
	public class UpdateTest
	{
		[SetUp]
		public void Setup()
		{
		}

		[Test]
		public void UpdateStoreShelvesById()
		{
			var options = new DbContextOptionsBuilder<PizzaDbContext>()
				.UseInMemoryDatabase(databaseName: "InMemoryDatabase")
				.Options;

			var db = new PizzaDbContext(options);

			db.StoreShelves
				.Set(x => x.Title, "123")
				.Where(x => x.Id == 1)
				.Update();

			//Assert.That(jsFile, Is.EqualTo("assets/luncher.ddee1e2b.js"));
		}
	}

}