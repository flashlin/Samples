﻿using Xunit;
using PreviewLibrary;
using System.Collections.Generic;
using ExpectedObjects;
using System.Linq;
using System.IO;
using Xunit.Abstractions;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using TestProject.Helpers;
using PreviewLibrary.RecursiveParser;

namespace TestProject
{
	public class SelectInnerJoinTest : SqlTestBase
	{
		public SelectInnerJoinTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void select_name_from_table1_inner_join_table2_on_tb2_id_eq_tb1_id()
		{
			var sql = @"SELECT name FROM user tb1 
Inner JOIN books tb2 WITH(nolock) 
ON tb2.id = tb1.id";

			var expr = new SqlParser().Parse(sql);
			@"SELECT name FROM user AS tb1
Inner JOIN books AS tb2 WITH(nolock)
ON tb2.id = tb1.id".ShouldEqual(expr);
		}

		[Fact]
		public void StarComment()
		{
			var sql = @"/*
123
*/";
			var expr = _sqlParser.Parse(sql);

			"".ShouldEqual(expr);
		}

		[Fact]
		public void select_join_all_select()
		{
			var sql = @"select 1
join all
select 2";

			var expr = _sqlParser.ParseSelectPartial(sql);

			@"SELECT 1
JOIN ALL
SELECT 2".ShouldEqual(expr);
		}

		[Fact]
		public void select_join_where_field_func_func_arithmetic()
		{
			var sql = @"select 1
	from customer
	join country c on c.id = b.id
	--Where createdDate = CAST(getdate()-1 as Date) 
";

			var expr = _sqlParser.ParseSelectPartial(sql);

			@"SELECT 1 FROM customer 
Inner JOIN country AS c ON c.id = b.id".ShouldEqual(expr);
		}


	}
}