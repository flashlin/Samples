using ExpectedObjects;
using PreviewLibrary;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using TestProject.Helpers;
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

			@"CREATE FUNCTION [dbo].[f1]
(
@a int,@b int,@c nvarchar(127)
)
RETURNS bit
AS BEGIN
SELECT 1
END".ShouldEqual(expr);
		}

		[Fact]
		public void create_func_returns_table()
		{
			var sql = @"create function a1( @b int )
returns @res table ( IsPersion bit, Reason nvarchar(100))
as
begin select 1 end";

			var expr = _sqlParser.Parse(sql);

			@"CREATE FUNCTION a1
(
@b int
)
RETURNS PreviewLibrary.DefineColumnTypeExpr
AS BEGIN
SELECT 1
END".ShouldEqual(expr);
		}
	}
}