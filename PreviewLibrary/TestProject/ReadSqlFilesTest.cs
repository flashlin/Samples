using PreviewLibrary;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class ReadSqlFilesTest : SqlTestBase
	{
		public ReadSqlFilesTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Theory]
		[InlineData(@"D:\Demo\1.sql")]
		public void TestSqlFile(string file)
		{
			var sql = File.ReadAllText(file);
			_sqlParser.ParseAll(sql).ToList();
		}

		[Fact]
		public void TestAllSqlFiles()
		{
			var sqlFolder = @"D:\VDisk\MyGitHub\SQL";
			var p = new SqlParser();
			var fileCount = 0;
			foreach (var sqlFile in ReadSqlFiles(sqlFolder))
			{
				var sql = File.ReadAllText(sqlFile);
				try
				{
					_outputHelper.WriteLine($"parsing='{sqlFile}'");
					p.ParseAll(sql).ToList();
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

		public IEnumerable<string> ReadSqlFiles(string folder)
		{
			var sqlFiles = Directory.EnumerateFiles(folder, "*.sql");
			var subDirs = Directory.EnumerateDirectories(folder);
			return sqlFiles.Concat(subDirs.SelectMany(x => ReadSqlFiles(x)));
		}
	}
}
