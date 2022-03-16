using Xunit;
using PreviewLibrary;
using System.Collections.Generic;
using ExpectedObjects;
using System.Linq;
using System.IO;
using Xunit.Abstractions;
using PreviewLibrary.Exceptions;

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
			var sql = @"select name from user tb1 
INNER JOIN books tb2 with(nolock) 
on tb2.id = tb1.id";

			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr
			{
				Fields = new List<SqlExpr>
				{
					new ColumnExpr
					{
						Name = "name"
					}
				},
				From = new TableExpr
				{
					Name = new IdentExpr { Name = "user" },
					AliasName = "tb1"
				},
				Joins = new []
				{
					new JoinExpr
					{
						Table = new TableExpr
						{
							Name = new IdentExpr { Name = "books" },
							AliasName = "tb2",
							WithOptions = new WithOptionsExpr
							{
								Options = new List<string>
								{
									"nolock"
								}
							}
						},
						JoinType = JoinType.Inner,
						Filter = new CompareExpr
						{
							Left = new IdentExpr
							{
								ObjectId = "tb2",
								Name = "id"
							},
							Oper = "=",
							Right = new IdentExpr
							{
								ObjectId = "tb1",
								Name = "id"
							}
						}
					}
				}.ToList()
			};

			expected.ToExpectedObject().ShouldEqual(expr);
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