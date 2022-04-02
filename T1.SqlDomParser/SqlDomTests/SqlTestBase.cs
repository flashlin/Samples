using T1.SqlDomParser;
using Xunit.Abstractions;

namespace SqlDomTests
{
	public class SqlTestBase
	{
		protected readonly ITestOutputHelper _outputHelper;
		protected readonly SqlParser _sqlParser;

		public SqlTestBase(ITestOutputHelper outputHelper)
		{
			this._outputHelper = outputHelper;
			_sqlParser = new SqlParser();
		}
	}
}