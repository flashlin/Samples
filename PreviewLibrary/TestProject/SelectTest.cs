using Xunit;
using PreviewLibrary;
using System.Linq;
using FluentAssertions;
using System.Collections.Generic;
using ExpectedObjects;
using Xunit.Abstractions;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using TestProject.Helpers;

namespace TestProject
{
	public class SelectTest : SqlTestBase
	{
		public SelectTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void select_1_from_tb1_tb2()
		{
			var sql = @"SELECT id FROM tb1,tb2";
			var expr = _sqlParser.ParseSelectPartial(sql);
			sql.ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_xxx_union_all_select_xxx()
		{
			var sql = @"WITH Cte1 (field1,field2)
AS (
	SELECT 1 FROM tb1
	UNION ALL
	SELECT 2 FROM tb2
)
SELECT field1,field2 FROM cte1";
			var expr = _sqlParser.ParseCtePartial(sql);

			@"WITH Cte1
(field1,field2)
AS (
SELECT 1 FROM tb1
UNION ALL
SELECT 2 FROM tb2
)".ShouldEqual(expr);
		}

		[Fact]
		public void select_var_eq_field_from_table()
		{
			var sql = "SELECT @id = id FROM customer";
			var expr = _sqlParser.ParseSelectPartial(sql);
			sql.ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_name()
		{
			var sql = "select name";
			var expr = new SqlParser().Parse(sql);

			"SELECT name".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_table_name()
		{
			var sql = "select tb1.name";
			var expr = new SqlParser().Parse(sql);

			"SELECT tb1.name".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_column1_column2()
		{
			var sql = "select id, name";
			var expr = new SqlParser().Parse(sql);

			"SELECT id,name".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_not_exists()
		{
			var sql = "select not exists(1)";
			var expr = Parse(sql);
			"SELECT NOT exists( 1 )".ShouldEqual(expr);
		}

		[Fact]
		public void select_1()
		{
			var sql = "select 1";
			var expr = new SqlParser().Parse(sql);

			"SELECT 1".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_count_star()
		{
			var sql = "select count(*)";
			var expr = _sqlParser.ParseSelectPartial(sql);

			"SELECT count( * )".ShouldEqual(expr);
		}

		[Fact]
		public void select_navigate_1()
		{
			var sql = "select - 1";
			var expr = new SqlParser().Parse(sql);

			"SELECT -1".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void if_func_select_where_is_null()
		{
			var sql = @"IF exists( SELECT 1 FROM customer WHERE name is NULL )
 BEGIN
		SELECT 1
 END";
			
			var expr = Parse(sql);

			@"IF exists( SELECT 1 FROM customer WHERE name IS NULL )
BEGIN
SELECT 1
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_a_eq_sum()
		{
			var sql = "select @a=sum(b) from tb1";
			var expr = _sqlParser.ParseSelectPartial(sql);
			"SELECT @a = sum( b ) FROM tb1".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void select_var_eq_a_or_b()
		{
			var sql = "select @a = @b | c from tb1";
			var expr = _sqlParser.ParseSelectPartial(sql);

			"SELECT @a = @b | c FROM tb1".ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
}