using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class MergeTest : TestBase
	{
		public MergeTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void merge()
		{
			var sql = @"merge into customer AS target
	using @other AS source ON target.Id = source.Id
	WHEN NOT MATCHED THEN
		INSERT (id, name) VALUES (source.Id, source.Name);";
		
			Parse(sql);

			ThenExprShouldBe(@"MERGE INTO customer AS Source
USING @other ON TARGET.Id = SOURCE.Id
WHEN NOT MATCHED
THEN
INSERT (id, name) VALUES(SOURCE.Id, SOURCE.Name)");
		}
	}
	}
