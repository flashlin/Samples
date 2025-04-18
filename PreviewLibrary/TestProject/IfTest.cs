﻿using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class IfTest : SqlTestBase
	{
		public IfTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void if_notEqual_expr()
		{
			var sql = @"if( isnull(@name, '') <> '' )
BEGIN select 1 END";

			var expr = _sqlParser.ParseIfPartial(sql);

			@"IF (isnull( @name,'' ) <> '')
BEGIN
SELECT 1
END".ShouldEqual(expr);
		}

		[Fact]
		public void if_var_not_in()
		{
			var sql = @"if @a not in (1,2)
BEGIN select 1 END";

			var expr = _sqlParser.ParseIfPartial(sql);

			@"IF @a NOT IN( 1,2 ) BEGIN SELECT 1 END".ShouldEqual(expr);
		}



		[Fact]
		public void if_begin_end_else_begin_end()
		{
			var sql = @"if @id = 1
begin select 2 end
else begin select 3 end";
			var expr = _sqlParser.ParseIfPartial(sql);
			@"IF @id = 1
BEGIN
SELECT 2
END
ELSE BEGIN
SELECT 3
END".ShouldEqual(expr);
		}

		[Fact]
		public void if_a_not_like_b()
		{
			var sql = @"IF N'a' NOT LIKE N'True'
BEGIN
	select 1
END";
			var expr = _sqlParser.ParseIfPartial(sql);
			@"IF N'a' NOT LIKE N'True'
BEGIN
SELECT 1
END".ShouldEqual(expr);
		}


		[Fact]
		public void if_not_func1_selectExpr_begin_selectExpr_end()
		{
			var sql = @"IF NOT exists(
    SELECT 1 FROM sys.databases
)
BEGIN
	SELECT 1
END";
			var expr = Parse(sql);
			@"IF NOT exists( SELECT 1 FROM sys.databases )
BEGIN
SELECT 1
END".ShouldEqual(expr);
		}

		[Fact]
		public void if_not_exists_selectExpr_begin_execExpr_end()
		{
			var sql = @"IF NOT exists(
    SELECT 1 FROM sys.databases
)
BEGIN
	EXEC my_func 'sa'
END";
			var expr = _sqlParser.Parse(sql);

			@"IF NOT exists( SELECT 1 FROM sys.databases )
BEGIN 
	EXEC my_func 'sa' 
END".ShouldEqual(expr);
		}


		[Fact]
		public void if_parentheses_arithmetic()
		{
			var sql = @"if (@a>=1 or @b + @c >=2)
BEGIN
	select 1
END";
			var expr = _sqlParser.ParseIfPartial(sql);

			@"IF (@a >= 1 or @b + @c >= 2)
BEGIN
SELECT 1
END".ShouldEqual(expr);
		}

		[Fact]
		public void if_condition_expr_else_if_condition_expr_else_expr()
		{
			var sql = @"
if a = 1
   set @a = 1
else if a = 2
   set @a = 2
else 
   set @a = 3
";
			var expr = _sqlParser.ParseIfPartial(sql);

			@"IF a = 1
SET @a = 1
ELSE IF a = 2
SET @a = 2
ELSE SET @a = 3".ShouldEqual(expr);
		}


		[Fact]
		public void if_select_condition()
		{
			var sql = @"
	if( select 1 from customer where a = CAST(@currTime as date) )
 
begin
	select 1
end
";
			var expr = _sqlParser.ParseIfPartial(sql);

			@"IF (SELECT 1 FROM customer WHERE a = CAST( @currTime AS date ))
BEGIN
SELECT 1
END".ShouldEqual(expr);
		}

		[Fact]
		public void if_else_if()
		{
			var sql = @"
	if a=1
	begin
		select 1
	end 
	else if b=2
	begin
		select 2
	end
";
			var expr = _sqlParser.ParseIfPartial(sql);

			@"IF a = 1 BEGIN 
	SELECT 1 
END 
ELSE IF b = 2 
BEGIN 
	SELECT 2
END".ShouldEqual(expr);
		}

		[Fact]
		public void if_return()
		{
			var sql = @"if @a=1 return";

			var expr = _sqlParser.ParseIfPartial(sql);

			@"IF @a = 1 RETURN".ShouldEqual(expr);
		}

		[Fact]
		public void if_return_func()
		{
			var sql = @"if @a=1 return isnull(@b,0)";

			var expr = _sqlParser.ParseIfPartial(sql);

			@"IF @a = 1 RETURN isnull( @b,0 )".ShouldEqual(expr);
		}

		[Fact]
		public void if_return_customFunc()
		{
			var sql = @"if @a=1 return aa(@b,0)";

			var expr = _sqlParser.ParseIfPartial(sql);

			@"IF @a = 1 RETURN aa( @b,0 )".ShouldEqual(expr);
		}


	}
}