using ExpectedObjects;
using PreviewLibrary;
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
			var sql = @"CREATE FUNCTION [dbo].[fAOSBetWin_6.7]
(
	@finalHomeScore int,
	@finalAwayScore int,	
	@betCondition nvarchar(127)
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
					Name = "[fAOSBetWin_6.7]",
					ObjectId = "[dbo]"
				},
				ArgumentsList = new List<List<ArgumentExpr>>
				{
					new List<ArgumentExpr> {
						new ArgumentExpr
						{
							Name = "@finalHomeScore",
							DataType = new DataTypeExpr
							{
								DataType = "int"
							}
						},new ArgumentExpr
						{
							Name = "@finalAwayScore",
							DataType = new DataTypeExpr
							{
								DataType = "int"
							}
						},
						new ArgumentExpr
						{
							Name = "@betCondition",
							DataType = new DataTypeExpr
							{
								DataType = "nvarchar",
								DataSize = new DataTypeSizeExpr
								{
									Size = 127
							  }
							}
						}
					}
				},
				ReturnDataType = new DataTypeExpr
				{
					DataType = "bit"
				},
				Body = new List<SqlExpr> {
					new SelectExpr
					{
						Fields = new List<SqlExpr> { 
							new IntegerExpr
							{
								Value = 1
							}
						}
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}