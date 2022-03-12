using Xunit;
using PreviewLibrary;
using ExpectedObjects;
using Xunit.Abstractions;

namespace TestProject
{
	public class InsertTest : SqlTestBase
	{
		public InsertTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void insert_table_fields_values_value1()
		{
			var sql = @"INSERT [dbo].[customer] ([id], [name], [lastname], [birth], [price]) VALUES 
				(267467, N'', NULL, CAST(0x0000A5E5006236FB AS DateTime), 1)";

			var expr = Parse(sql);

			new InsertValuesExpr
			{

			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}