using PreviewLibrary;
using Xunit;
using Xunit.Abstractions;
using ExpectedObjects;
using System.Collections.Generic;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;

namespace TestProject
{

	public class DeleteTest : SqlTestBase
	{
		public DeleteTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void delete_from_table_where_id_in_selectExpr()
		{
			var sql = "delete from customer where id in (select pid from products)";
			var expr = Parse(sql);

			new DeleteExpr
			{
				Table = new IdentExpr
				{
					Name = "customer"
				},
				WhereExpr = new CompareExpr
				{
					Left = new IdentExpr
					{
						Name = "id"
					},
					Oper = "in",
					Right = new GroupExpr
					{
						Expr = new SelectExpr
						{
							Fields = new SqlExprList
							{
								Items = new List<SqlExpr>
								{
									new ColumnExpr
									{
										Name = "pid"
									}
								}
							},
							From = new TableExpr
							{
								Name = new IdentExpr
								{
									Name = "products"
								}
							}
						}
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}