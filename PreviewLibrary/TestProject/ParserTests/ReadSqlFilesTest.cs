using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace TestProject.ParserTests
{
	public class ReadSqlFilesTest : ParserTestBase
	{
		public ReadSqlFilesTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void TestAllSqlFiles()
		{
			var sqlFolder = @"D:\VDisk\MyGitHub\SQL";
			var fileCount = 0;
			foreach (var sqlFile in ReadSqlFiles(sqlFolder))
			{
				var sql = File.ReadAllText(sqlFile);
				try
				{
					Parse(sql);
				}
				catch
				{
					_outputHelper.WriteLine($"parsedCount={fileCount}");
					_outputHelper.WriteLine($"'{sqlFile}'");
					throw;
				}
				fileCount++;
			}
			_outputHelper.WriteLine($"Total parsed Count={fileCount}");
			_outputHelper.WriteLine($"=== END ===");
		}

		protected IEnumerable<string> ReadSqlFiles(string folder)
		{
			var sqlFiles = Directory.EnumerateFiles(folder, "*.sql");
			var subDirs = Directory.EnumerateDirectories(folder);
			return sqlFiles.Concat(subDirs.SelectMany(x => ReadSqlFiles(x)));
		}
	}
}
