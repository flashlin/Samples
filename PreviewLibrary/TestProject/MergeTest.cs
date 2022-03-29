using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class MergeTest : SqlTestBase
	{
		public MergeTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void merge_into()
		{
			var sql = @"MERGE INTO customer AS target
				USING @tmpCustomer AS source ON target.id = source.id
				WHEN NOT MATCHED THEN
				INSERT (id, name) VALUES (source.id, source.name)";

			var expr = _sqlParser.Parse(sql);

			@"MERGE INTO customer target
ON @tmpCustomer source
WHEN NOT MATCHED
THEN
INSERT (id,name) VALUES(source.id,source.name)".ShouldEqual(expr);
		}
	}

}