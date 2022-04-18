using Xunit.Abstractions;
using Xunit;
using System.IO;

namespace TestProject.PrattTests
{
	public class ReadSqlFileTest : TestBase
	{
		public ReadSqlFileTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void read_sql_file()
		{
			var sqlFile = @"D:\demo\1.sql";
			ReadSqlFile(0, sqlFile);
		}


		[Fact]
		public void read_all_sql_files()
		{
			var sqlFolder = @"D:\VDisk\MyGitHub\SQL";
			var fileCount = 0;
			foreach (var sqlFile in ReadSqlFiles(sqlFolder))
			{
				ReadSqlFile(fileCount, sqlFile);
				fileCount++;
			}
			_outputHelper.WriteLine($"Total parsed Count={fileCount}");
			_outputHelper.WriteLine($"=== END ===");
		}

		private void ReadSqlFile(int fileCount, string sqlFile)
		{
			var sql = File.ReadAllText(sqlFile);
			try
			{
				ParseAll(sql);
			}
			catch
			{
				_outputHelper.WriteLine($"parsedCount={fileCount}");
				_outputHelper.WriteLine($"'{sqlFile}'");
				throw;
			}
		}
	}
}
