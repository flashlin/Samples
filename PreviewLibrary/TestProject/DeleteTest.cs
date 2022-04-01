using PreviewLibrary;
using Xunit;
using Xunit.Abstractions;
using ExpectedObjects;
using System.Collections.Generic;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using TestProject.Helpers;

namespace TestProject
{

	public class DeleteTest : SqlTestBase
	{
		public DeleteTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void delete_from_table_where_field_in_selectExpr()
		{
			var sql = "DELETE FROM customer WHERE id IN (SELECT pid FROM products)";
			var expr = Parse(sql);
			sql.ShouldEqual(expr);
		}

		[Fact]
		public void delete_from_table_where_field_in()
		{
			var sql = @"delete from customer
where id in (
select id from customer where loginname like 'abc%'
)";
			var expr = _sqlParser.ParseDeletePartial(sql);
			"DELETE FROM customer WHERE id IN (SELECT id FROM customer WHERE loginname LIKE 'abc%')".ShouldEqual(expr);
		}

		[Fact]
		public void delete_table_where_field_eq_var()
		{
			var sql = @"DELETE customer WHERE id=@testId";
			var expr = _sqlParser.ParseDeletePartial(sql);

			"DELETE FROM customer WHERE id = @testId".ShouldEqual(expr);
		}

		[Fact]
		public void delete_from_table_output_x1_x2_into_table_columns_where()
		{
			var sql = @"delete from customer
				output deleted.id, GETDATE()
				into otherCustomer([id],[ModifiedOn])
				where id = @id";
			var expr = _sqlParser.ParseDeletePartial(sql);

			@"DELETE FROM customer
OUTPUT deleted.id,GETDATE() 
INTO otherCustomer([id],[ModifiedOn]) 
WHERE id = @id".ShouldEqual(expr);
		}

		[Fact]
		public void delete_from_table_with_rowlock_where()
		{
			var sql = @"delete from customer with(rowlock) where id=@id";

			var expr = _sqlParser.ParseDeletePartial(sql);

			@"DELETE FROM customer WITH(rowlock) WHERE id = @id".ShouldEqual(expr);
		}

		[Fact]
		public void delete_top_var_from_table()
		{
			var sql = @"delete top(@a) from customer";

			var expr = _sqlParser.ParseDeletePartial(sql);

			@"DELETE TOP(@a) FROM customer".ShouldEqual(expr);
		}
	}
}