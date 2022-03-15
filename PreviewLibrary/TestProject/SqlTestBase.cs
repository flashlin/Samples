using PreviewLibrary;
using Xunit.Abstractions;

namespace TestProject
{
	public abstract class SqlTestBase
	{

		protected readonly ITestOutputHelper _outputHelper;
		protected SqlParser _sqlParser;

		public SqlTestBase(ITestOutputHelper outputHelper)
		{
			_sqlParser = new SqlParser();
			this._outputHelper = outputHelper;
		}

		protected SqlExpr Parse(string sql)
		{
			return _sqlParser.Parse(sql);
		}
	}
}