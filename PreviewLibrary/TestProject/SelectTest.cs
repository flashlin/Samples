using Xunit;
using PreviewLibrary;
using System.Linq;
using FluentAssertions;
using System.Collections.Generic;
using ExpectedObjects;
using Xunit.Abstractions;

namespace TestProject
{
	public class SelectTest : SqlTestBase
	{
		public SelectTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void select_name()
		{
			var sql = "select name";
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
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_table_name()
		{
			var sql = "select tb1.name";
			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr()
			{
				Fields = new List<SqlExpr>
				{
					new ColumnExpr
					{
						Table = "tb1",
						Name = "name"
					}
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_column1_column2()
		{
			var sql = "select id, name";
			var expr = new SqlParser().Parse(sql);

			var expected = new SelectExpr()
			{
				Fields = new List<SqlExpr>
				{
					new ColumnExpr
					{
						Name = "id"
					},
					new ColumnExpr
					{
						Name = "name"
					}
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_1()
		{
			var sql = "select 1";
			var expr = new SqlParser().Parse(sql);

			var expected =new SelectExpr()
			{
				Fields = new List<SqlExpr>
				{
					new IntegerExpr
					{
						Value = 1
					}
				},
			};

			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void if_func_select_where_is_null()
		{
			var sql = @"if exists (select 1 from customer where name is null)
 Begin
		select 1
 End";
			var expr = Parse(sql);
			new IfExpr()
			{
				Condition = new SqlFuncExpr
				{
					Name = "exists",
					Arguments = new SqlExpr[] { 
						new SelectExpr
						{
							Fields = new List<SqlExpr> { 
								new IntegerExpr
								{
									Value = 1
								}
							},
							From = new TableExpr
							{
								Name = new IdentExpr
								{
									 Name = "customer"
								}
							},
							WhereExpr = new CompareExpr
							{
								Left = new IdentExpr
								{
									 Name = "name"
								},
								Oper = "is",
								Right = new NullExpr
								{
									 Token = "null"
								}
							}
						}
					}
				},
				Body = new List<SqlExpr> { 
					new SelectExpr
					{
						Fields = new List<SqlExpr> { 
							new IntegerExpr
							{
								Value = 1
							}
						}
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}