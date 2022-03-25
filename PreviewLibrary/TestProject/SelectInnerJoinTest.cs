using Xunit;
using PreviewLibrary;
using System.Collections.Generic;
using ExpectedObjects;
using System.Linq;
using System.IO;
using Xunit.Abstractions;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;

namespace TestProject
{
	public class SelectInnerJoinTest
	{
		private readonly ITestOutputHelper outputHelper;

		public SelectInnerJoinTest(ITestOutputHelper outputHelper)
		{
			this.outputHelper = outputHelper;
		}

		[Fact]
		public void select_name_from_table1_inner_join_table2_on_tb2_id_eq_tb1_id()
		{
			var sql = @"SELECT name FROM user tb1 
Inner JOIN books tb2 WITH(nolock) 
ON tb2.id = tb1.id";

			var expr = new SqlParser().Parse(sql);
			@"SELECT name FROM user AS tb1
Inner JOIN books as tb2 WITH(nolock)
ON tb2.id = tb1.id".MergeToCode().ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void StarComment()
		{
			var sql = @"/*
123
*/";
			var expr = new SqlParser().Parse(sql);
			var expected = new CommentExpr
			{
				Text = sql
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}
	}
}