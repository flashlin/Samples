using System.IO;
using T1.SqlLocalData;
using Xunit;

namespace SqlLocalDataTests
{
	public class CreateTest
	{
		[Fact]
		public void create_database()
		{
			var databaseFile = @"D:\Demo\test.mdf";
			var localDb = new SqlLocalDb();
			localDb.CreateDatabase(databaseFile);
			Assert.True(File.Exists(databaseFile));
			//File.Delete(databaseFile);
		}
	}
}