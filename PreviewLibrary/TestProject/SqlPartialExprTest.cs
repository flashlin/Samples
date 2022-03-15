using ExpectedObjects;
using PreviewLibrary;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class SqlPartialExprTest : SqlTestBase
	{
		public SqlPartialExprTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void case_when()
		{
			var sql = "CASE WHEN @ExchangeRate = -1 THEN [ExchangeRate] ELSE @ExchangeRate END";
			var expr = _sqlParser.ParseCasePartial(sql);
			new CaseExpr
         {
            WhenList = new List<WhenThenExpr> { 
               new WhenThenExpr
					{
						 When = new CompareExpr
						 {
							  Left = new IdentExpr
							  {
									Name = "@ExchangeRate"
							  },
							  Oper = "=",
							  Right = new IntegerExpr
							  {
									Value = -1
							  }
						 },
						 Then = new IdentExpr
						 {
							  Name = "[ExchangeRate]"
						 }
					}
				},
            Else = new IdentExpr
            {
               Name = "@ExchangeRate"
            }
         }.ToExpectedObject().ShouldEqual(expr);
		}
	}
}