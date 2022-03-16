using ExpectedObjects;
using PreviewLibrary;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
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

		[Fact]
		public void table()
		{
			var sql = @"table
(
	BetOption nvarchar(10)
)";
			var expr = _sqlParser.ParseDataTypePartial(sql);
			new TableTypeExpr
			{
				ColumnTypeList = new List<SqlExpr> 
				{ 
					new DefineColumnTypeExpr
					{
						 Name = new IdentExpr
						 {
							  Name = "BetOption"
						 },
						 DataType = new DataTypeExpr
						 {
							  DataType = "nvarchar",
							  DataSize = new DataTypeSizeExpr
							  {
									Size = 10
							  }
						 }
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}