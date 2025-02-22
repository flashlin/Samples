﻿using ExpectedObjects;
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
		public void create_procedure_arg_as_datatype()
		{
			var sql = @"create PROCEDURE [dbo].[f1]
	@a as int
AS
BEGIN
	select 1
END";
			var expr = _sqlParser.ParseCreateSpPartial(sql);

			@"CREATE PROCEDURE [dbo].[f1] @a int AS BEGIN SELECT 1 END".ShouldEqual(expr);
		}

		[Fact]
		public void create_procedure_arg_as_datatype_output()
		{
			var sql = @"create PROCEDURE [dbo].[f1]
	@a as int output
AS
BEGIN
	select 1
END";
			var expr = _sqlParser.ParseCreateSpPartial(sql);

			@"CREATE PROCEDURE [dbo].[f1] 
@a int OUTPUT 
AS BEGIN SELECT 1 END".ShouldEqual(expr);
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
RETURNS @res TABLE (
	IsPersion bit
	,Reason nvarchar(100)
)
AS BEGIN
	SELECT 1
END".ShouldEqual(expr);
		}

		[Fact]
		public void create_partition_function()
		{
			var sql = @"CREATE PARTITION FUNCTION [a](INT)
    AS RANGE
    FOR VALUES (0, 1, 3)";

			var expr = _sqlParser.Parse(sql);

			@"CREATE PARTITION FUNCTION [a]( INT )
AS RANGE
FOR VALUES (
0,1,3
)".ShouldEqual(expr);
		}

		[Fact]
		public void create_partition_scheme()
		{
			var sql = @"CREATE PARTITION SCHEME [a]
				AS PARTITION [b]
				TO ([PRIMARY], [PRIMARY])";

			var expr = _sqlParser.Parse(sql);

			@"CREATE PARTITION SCHEME [a]
AS PARTITION [b]
TO (
[PRIMARY],[PRIMARY]
)".ShouldEqual(expr);
		}


		[Fact]
		public void create_procedure_args_comment()
		{
			var sql = @"create procedure xx
@name varchar(50) /* test */
as
begin
	select 1
end";

			var expr = _sqlParser.ParseCreateSpPartial(sql);

			@"CREATE PROCEDURE xx
@name varchar(50)
AS
BEGIN
SELECT 1
END".ShouldEqual(expr);
		}

	[Fact]
		public void create_procedure_single_body()
		{
			var sql = @"create procedure xx
@name varchar(50) 
as
	select 1
";

			var expr = _sqlParser.ParseCreateSpPartial(sql);

			@"CREATE PROCEDURE xx
@name varchar(50)
AS
BEGIN
SELECT 1
END".ShouldEqual(expr);
		}

		[Fact]
		public void create_procedure_as_begin_tran()
		{
			var sql = @"create procedure xx
@name varchar(50) 
as
	begin tran
	rollback transaction
";

			var expr = _sqlParser.ParseCreateSpPartial(sql);

			@"CREATE PROCEDURE xx @name varchar(50) 
AS BEGIN 
	BEGIN TRANSACTION 
	ROLLBACK TRANSACTION 
END".ShouldEqual(expr);
		}


		
	}
}