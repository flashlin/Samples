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

            ThenExprShouldBe(@"MERGE INTO customer AS target
USING @other AS source ON TARGET.Id = SOURCE.Id
WHEN NOT MATCHED
THEN
INSERT (id, name) VALUES(SOURCE.Id, SOURCE.Name)");
        }

        [Fact]
        public void merge_target1_target2()
        {
            var sql = @"MERGE customer AS target  
    USING #tmoCustomer  AS source ON (target.id = source.id)  
    WHEN MATCHED and target.tstamp < source.tstamp and target.status in ('waiting')
		THEN   
			UPDATE SET   
				Target.[birth]=Source.[birth],  
				Target.[addr]=Source.[addr]  
		WHEN NOT MATCHED by Target THEN  
			INSERT ([id],[name]) VALUES(source.id,source.name);";

            Parse(sql);

            ThenExprShouldBe(@"MERGE customer AS target
USING #tmoCustomer AS source ON ( TARGET.id = SOURCE.id )
WHEN MATCHED AND TARGET.tstamp < SOURCE.tstamp AND TARGET.status IN ('waiting')
THEN
UPDATE SET Target.[birth] = SOURCE.[birth]	,
Target.[addr] = SOURCE.[addr]
WHEN NOT MATCHED BY TARGET
THEN
INSERT ([id], [name]) VALUES(SOURCE.id, SOURCE.name)");
        }

        [Fact]
        public void merge_using()
        {
            var sql = @"MERGE customer 
	USING ( VALUES (@id, @name)) 
	AS S (id, name)
	ON T.Id = S.id
	WHEN MATCHED THEN UPDATE
		SET T.name = S.name
	WHEN NOT MATCHED THEN 
		INSERT VALUES (S.id, S.name);";

            Parse(sql);

            ThenExprShouldBe(@"MERGE customer 
USING ( VALUES (@id, @name) ) AS S(id, name) ON T.Id = S.id
WHEN MATCHED THEN 
	UPDATE SET T.name = S.name
WHEN NOT MATCHED THEN
	INSERT VALUES(S.id, S.name)");
        }

        [Fact]
        public void merge_when_matched()
        {
            var sql = @"merge customer as tar
using @customer as src
	on tar.id = src.id 
	when not matched then
	    insert(id, name) 
		values(src.id, src.name)
	when matched then
	    update set name = name + 'a',
	        price = tar.price + src.price
     ;";

            Parse(sql);

            ThenExprShouldBe(@"MERGE customer ");
        }
    }
}