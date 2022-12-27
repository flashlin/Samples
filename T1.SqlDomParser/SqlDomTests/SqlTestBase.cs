using FluentAssertions;
using T1.SqlDom.Expressions;
using Xunit.Abstractions;

namespace SqlDomTests
{
    public class SqlTestBase
    {
        private readonly ITestOutputHelper _outputHelper;
        private readonly SqlParser _sqlParser;
        private SqlExpr _result = null!;

        protected SqlTestBase(ITestOutputHelper outputHelper)
        {
            _outputHelper = outputHelper;
            _sqlParser = new SqlParser();
        }

        protected void Parse(string sql)
        {
            _result = _sqlParser.Parse(sql);
        }

        protected void ThenResultShouldBe(string expected)
        {
            _result.ToSqlString().Should().Be(expected);
        }
    }
}