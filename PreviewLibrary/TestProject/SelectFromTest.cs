using Xunit;
using PreviewLibrary;
using FluentAssertions;
using System.Collections.Generic;
using FluentAssertions.Equivalency;
using ExpectedObjects;
using System.Linq;

namespace TestProject
{
	public class SelectFromTest
	{
		[Fact]
		public void select_column_from_table()
		{
			var sql = "select name from user";
			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr()
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
					Name = new IdentExpr 
					{ 
						Name = "user"
					}
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_column_from_brackets_table()
		{
			var sql = "select name from [user]";
			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr()
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
					Name = new IdentExpr 
					{ 
						Name = "[user]"
					}
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_column_aliasName_from_table()
		{
			var sql = "select name n1 from user";
			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr()
			{
				Fields = new List<SqlExpr>
				{
					new ColumnExpr
					{
						Name = "name",
						AliasName = "n1",
					}
				},
				From = new TableExpr
				{
					Name = new IdentExpr
					{
						Name = "user"
					}
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_column_as_aliasName_from_table()
		{
			var sql = "select name as n1 from user";
			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr()
			{
				Fields = new List<SqlExpr>
				{
					new ColumnExpr
					{
						Name = "name",
						AliasName = "n1",
					}
				},
				From = new TableExpr
				{
					Name = new IdentExpr 
					{ 
						Name = "user" 
					}
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_column_as_aliasName_from_table_aliasName()
		{
			var sql = "select name as n1 from user tb1";
			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr()
			{
				Fields = new List<SqlExpr>
				{
					new ColumnExpr
					{
						Name = "name",
						AliasName = "n1",
					}
				},
				From = new TableExpr
				{
					Name = new IdentExpr { Name = "user" },
					AliasName = "tb1"
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_column_as_aliasName_from_table_as_aliasName()
		{
			var sql = "select name as n1 from user as tb1";
			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr()
			{
				Fields = new List<SqlExpr>
				{
					new ColumnExpr
					{
						Name = "name",
						AliasName = "n1",
					}
				},
				From = new TableExpr
				{
					Name = new IdentExpr { Name = "user" },
					AliasName = "tb1"
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_column_as_aliasName_from_table_as_aliasName_with_nolock()
		{
			var sql = "select name as n1 from user as tb1 with(nolock)";
			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr()
			{
				Fields = new List<SqlExpr>
				{
					new ColumnExpr
					{
						Name = "name",
						AliasName = "n1",
					}
				},
				From = new TableExpr
				{
					Name = new IdentExpr { Name = "user" },
					AliasName = "tb1",
					WithOptions = new WithOptionsExpr
					{
						Options = new List<string>
						{
							"nolock"
						}
					}
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
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

			var expected = new SelectExpr()
			{
				Fields = new List<SqlExpr>
				{
					new ColumnExpr
					{
						Name = "CustID"
					},
					new ColumnExpr
					{
						Name = "Transid"
					},
					new ColumnExpr
					{
						Name = "TransDate"
					},
				},
				From = new TableExpr
				{
					Name = new IdentExpr { Name = "Statement" },
					WithOptions = new WithOptionsExpr
					{
						Options = new List<string>
						{
							"nolock"
						}
					}
				},
				WhereExpr = new AndOrExpr
				{
					Left = new AndOrExpr
					{
						Left = new LikeExpr
						{
							Left = new IdentExpr
							{
								Name = "TransDesc"
							},
							Right = "'Full Transfer%'"
						},
						Oper = "and",
						Right = new CompareExpr
						{
							Left = new IdentExpr
							{
								Name = "TransDate1"
							},
							Oper = ">=",
							Right = new IdentExpr
							{
								Name = "@from"
							},
						}
					},
					Oper = "or",
					Right = new CompareExpr
					{
						Left = new IdentExpr
						{
							Name = "TransDate2"
						},
						Oper = "<",
						Right = new IdentExpr
						{
							Name = "@to"
						}
					}
				}
			};

			//expr.Should().BeEquivalentTo(expected, o => o.RespectingRuntimeTypes());
			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void Select2()
		{
			var sql = @"select 1
select 2";
			var exprs = new SqlParser().ParseAll(sql).ToList();

			var expected = new List<SqlExpr>
			{
				new SelectExpr()
				{
					Fields = new List<SqlExpr>
					{
						new IntegerExpr
						{
								Value = 1
						}
					}
				}, 
				new SelectExpr()
				{
					Fields = new List<SqlExpr>
					{
						new IntegerExpr
						{
							Value = 2
						}
					}
				}
			};

			expected.ToExpectedObject().ShouldEqual(exprs);
		}
	}
}