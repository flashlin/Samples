﻿using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class WhereTest : SqlTestBase
	{
		public WhereTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void where_a_and_b_eq_c()
		{
			var sql = "where a & b = c";
			var expr = _sqlParser.ParseWherePartial(sql);
			"a & b = c".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void where_a_is_null()
		{
			var sql = "where a is null";
			var expr = _sqlParser.ParseWherePartial(sql);
			"a IS NULL".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void a_like_b_and_or()
		{
			var sql = "WHERE desc LIKE 'a%' and b >= @c or b < @d";
			var expr = _sqlParser.ParseWherePartial(sql);
			"desc LIKE 'a%' and b >= @c or b < @d".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void a_in_select()
		{
			var sql = "WHERE id IN (SELECT pid FROM products)";
			var expr = _sqlParser.ParseWherePartial(sql);
			"id IN (SELECT pid FROM products)".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void where_field_eq_cast_add_cast()
		{
			var sql = "WHERE name = CAST( @a AS nvarchar(3) ) + ':' + CAST( @b AS nvarchar(3) )";
			var expr = _sqlParser.ParseWherePartial(sql);
			sql.ToExpectedObject().ShouldEqual("WHERE " + expr.ToString());
		}

		[Fact]
		public void where_field_between()
		{
			var sql = "WHERE id between 1 and 8";
			var expr = _sqlParser.ParseWherePartial(sql);
			"id BETWEEN 1 AND 8".ShouldEqual(expr);
		}

		[Fact]
		public void where_field_between_or_in()
		{
			var sql = @"WHERE a > 1 and (b between 1 and 10 or c in (1, 2, 3))";
			var expr = _sqlParser.ParseWherePartial(sql);
			"a > 1 and (b BETWEEN 1 AND 10 or c IN (1,2,3))".ShouldEqual(expr);
		}

		[Fact]
		public void where_field_p()
		{
			var sql = @"WHERE id=1 and 
   (
		(@a - @b) + (@c - @d) + 
		(@b1 -b2) + (b3 - b4)
	)";
			var expr = _sqlParser.ParseWherePartial(sql);
			"id = 1 and ((@a - @b) + (@c - @d) + (@b1 - b2) + (b3 - b4))".ShouldEqual(expr);
		}

		
	}
}