using ExpectedObjects;
using PreviewLibrary;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class CreateTest : SqlTestBase
	{
		public CreateTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void create_function()
		{
			var sql = @"CREATE FUNCTION [dbo].[f1]
(
	@a int,
	@b int,	
	@c nvarchar(127)
)
RETURNS bit
AS
BEGIN
	select 1
END";
			var expr = Parse(sql);
			new CreateFunctionExpr
			{
				Name = new IdentExpr
				{
					Name = "[f1]",
					ObjectId = "[dbo]"
				},
				Arguments = CreateSqlExprList(
					new ArgumentExpr
					{
						Name = "@a",
						DataType = new DataTypeExpr
						{
							DataType = "int"
						}
					},new ArgumentExpr
					{
						Name = "@b",
						DataType = new DataTypeExpr
						{
							DataType = "int"
						}
					},
					new ArgumentExpr
					{
						Name = "@c",
						DataType = new DataTypeExpr
						{
							DataType = "nvarchar",
							DataSize = new DataTypeSizeExpr
							{
								Size = 127
						  }
						}
					}
				),
				ReturnDataType = new DataTypeExpr
				{
					DataType = "bit"
				},
				Body = new List<SqlExpr> {
					new SelectExpr
					{
						Fields = CreateSqlExprList(
							new IntegerExpr
							{
								Value = 1
							}
						)
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}