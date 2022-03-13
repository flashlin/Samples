using PreviewLibrary;
using Xunit.Abstractions;

namespace TestProject
{
	public abstract class SqlTestBase
	{
		protected readonly ITestOutputHelper _outputHelper;

		public SqlTestBase(ITestOutputHelper outputHelper)
		{
			this._outputHelper = outputHelper;
		}

		protected SqlExpr Parse(string sql)
		{
			return new SqlParser().Parse(sql);
		}
	}
}