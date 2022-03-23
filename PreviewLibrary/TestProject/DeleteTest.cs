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

		[Fact]
		public void delete_from_tb1_where_id_in()
		{
			var sql = @"delete from customer
where id in (
select id from customer where loginname like 'abc%'
)";
			var expr = _sqlParser.ParseDeletePartial(sql);
			"DELETE FROM customer WHERE id IN (SELECT id FROM customer WHERE loginname LIKE 'abc%')".ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
}