using System;
using System.IO;
using T1.SqlLocalData;
using Xunit;

namespace SqlLocalDataTests
{
	public class CreateTest : IDisposable
	{
		public CreateTest()
		{
			
		}

		public void Dispose()
		{
		}

		[Fact]
		public void create_database()
		{
			var instanceName = "localtest";
			var databaseFile = @"D:\Demo\test.mdf";
			var localDb = new SqlLocalDb();
			localDb.CreateInstance(instanceName);
			localDb.CreateDatabase(instanceName, databaseFile);
			Assert.True(File.Exists(databaseFile));
			//File.Delete(databaseFile);
		}
	}
}