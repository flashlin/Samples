using Xunit;
using PreviewLibrary;
using FluentAssertions;
using System.Collections.Generic;
using FluentAssertions.Equivalency;
using ExpectedObjects;
using System.Linq;
using PreviewLibrary.Expressions;
using Xunit.Abstractions;
using TestProject.Helpers;

namespace TestProject
{
	public class SelectFromTest : SqlTestBase
	{
		public SelectFromTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void select_1_from_table_where_cast_add_cast()
		{
			var sql = "SELECT 1 FROM @a WHERE name = CAST( @b AS nvarchar(3) ) + ':' + CAST( @c AS nvarchar(3) )";
			var expr = _sqlParser.ParseSelectPartial(sql);
			sql.ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_column_from_table()
		{
			var sql = "select name from user";
			var expr = new SqlParser().Parse(sql);
			"SELECT name FROM user".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_column_from_brackets_table()
		{
			var sql = "select name from [user]";
			var expr = new SqlParser().Parse(sql);
			"SELECT name FROM [user]".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_column_aliasName_from_table()
		{
			var sql = "SELECT name n1 FROM user";
			var expr = _sqlParser.ParseSelectPartial(sql);
			"SELECT name as n1 FROM user".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_column_as_aliasName_from_table()
		{
			var sql = "select name as n1 from user";
			var expr = new SqlParser().Parse(sql);
			"SELECT name as n1 FROM user".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_column_as_aliasName_from_table_aliasName()
		{
			var sql = "SELECT name as n1 FROM user tb1";
			var expr = new SqlParser().Parse(sql);
			"SELECT name as n1 FROM user AS tb1".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_column_as_aliasName_from_table_as_aliasName()
		{
			var sql = "SELECT name as n1 FROM user AS tb1";
			var expr = new SqlParser().Parse(sql);
			sql.ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_column_as_aliasName_from_table_as_aliasName_with_nolock()
		{
			var sql = "SELECT name as n1 FROM user AS tb1 WITH(nolock)";
			var expr = new SqlParser().Parse(sql);
			sql.ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_fields_from_table_nolock_where_like_and_greaterThan_or_smallerThan()
		{
			var sql = @"select CustID ,Transid, TransDate
	from Statement with (nolock)
	where TransDesc like 'Full Transfer%'
		and TransDate1 >= @from
		or TransDate2 < @to";

			var expr = new SqlParser().Parse(sql) as SelectExpr;
			@"SELECT CustID,Transid,TransDate
FROM Statement WITH(nolock)
WHERE TransDesc LIKE 'Full Transfer%'
and TransDate1 >= @from
or TransDate2 < @to".MergeToCode().ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_variable_eq_1_from_table()
		{
			var sql = "select @id = 1 from tb1";
			var expr = _sqlParser.ParseSelectPartial(sql);
			"SELECT @id = 1 FROM tb1".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void Select2()
		{
			var sql = @"select 1
select 2";
			var exprs = new SqlParser().ParseAll(sql).ToList();

			var exprsCode = string.Join("\r\n", exprs.Select(x => $"{x}"));

			@"SELECT 1
SELECT 2".ToExpectedObject().ShouldEqual(exprsCode);
		}

		[Fact]
		public void select_variableName_eq_func1_from_tb1()
		{
			var sql = @"select @a = round((@b -1), 0) from tb1";
			
			var expr = _sqlParser.ParseSelectPartial(sql);

			"SELECT @a = round( @b - 1,0 ) FROM tb1".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_1_from_group_select_2()
		{
			var sql = "select 1 from (select 2)";
			var expr = _sqlParser.ParseSelectPartial(sql);

			"SELECT 1 FROM (SELECT 2)".ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_group_by()
		{
			var sql = "select 1 from tb1 group by id, name";
			var expr = _sqlParser.ParseSelectPartial(sql);
			@"SELECT 1 FROM tb1
	GROUP BY id,name".ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_table_where_name_eq_func0_and_func1_eq_string()
		{
			var sql = @"SELECT 1 FROM sys.databases WHERE name = DB_NAME() AND SUSER_SNAME( owner_sid ) = 'sa'";
			var expr = Parse(sql);
			sql.ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_1_from_table_where_name_eq_func0()
		{
			var sql = @"SELECT 1 FROM sys.databases WHERE name = DB_NAME()";
			var expr = Parse(sql);
			sql.ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
}
