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
			var sql = "DELETE FROM customer WHERE id IN (SELECT pid FROM products)";
			var expr = Parse(sql);
			sql.ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
}