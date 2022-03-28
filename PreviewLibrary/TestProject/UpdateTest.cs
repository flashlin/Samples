using ExpectedObjects;
using PreviewLibrary;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class UpdateTest : SqlTestBase
	{
		public UpdateTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void update_table_set_field_eq_field_add_1()
		{
			var sql = "Update customer set price = rate + 1";
			var expr = Parse(sql);
			@"UPDATE $customer
SET price = rate + 1".ShouldEqual(expr);
		}

		[Fact]
		public void update_table_set_field_eq_variable()
		{
			var sql = "UPDATE [dbo].[TracDelay] SET [Status] = @Status WHERE[Id] = @Id";
			var expr = Parse(sql);
			@"UPDATE $[dbo].[TracDelay]
SET [Status] = @Status
WHERE [Id] = @Id".ShouldEqual(expr);
		}

		[Fact]
		public void update_set_case_when()
		{
			var sql = @"UPDATE [dbo].[TracDelay] 
SET [ExchangeRate] = CASE WHEN @ExchangeRate = -1 THEN [ExchangeRate] ELSE @ExchangeRate END";
			var expr = _sqlParser.ParseUpdatePartial(sql);

			@"UPDATE $[dbo].[TracDelay]
SET [ExchangeRate] = CASE
	WHEN @ExchangeRate = -1 THEN [ExchangeRate]
	ELSE @ExchangeRate
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void update_table_with_nolock()
		{
			var sql = @"UPDATE customer WITH(ROWLOCK)
	SET name=@name, birth=@birth
	WHERE	id=@id";

			var expr = _sqlParser.ParseUpdatePartial(sql);

			@"UPDATE $customer WITH(ROWLOCK)
SET name = @name,birth = @birth
WHERE id = @id".ShouldEqual(expr);
		}
	}
}