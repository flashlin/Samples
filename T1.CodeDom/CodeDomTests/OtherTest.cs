using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
    public class OtherTest : TestBase
    {
        public OtherTest(ITestOutputHelper outputHelper) : base(outputHelper)
        {
        }

        [Fact]
        public void multiComment()
        {
            var sql = @"/*
123
*/";
            Parse(sql);

            ThenExprShouldBe(@"/* 
123
*/");
        }

        [Fact]
        public void script_on_error_exit()
        {
            var sql = ":on error exit";

            Parse(sql);

            ThenExprShouldBe(":ON ERROR EXIT");
        }

        [Fact]
        public void identifier_not_like_var()
        {
            var sql = "name not like @name";

            Parse(sql);

            ThenExprShouldBe("name NOT LIKE @name");
        }

        [Fact]
        public void grant_connect_to_user()
        {
            var sql = "grant connect to [user_Name]";

            Parse(sql);

            ThenExprShouldBe("GRANT CONNECT TO [user_Name]");
        }

        [Fact]
        public void begin_tran()
        {
            var sql = "begin tran";

            Parse(sql);

            ThenExprShouldBe("BEGIN TRANSACTION");
        }

        [Fact]
        public void pivot()
        {
            var sql = "pivot (max(id) for idType in( [4], [3], [2] ) ) piv";

            Parse(sql);

            ThenExprShouldBe("PIVOT(MAX( id ) FOR idType IN ([4], [3], [2])) AS piv");
        }


        [Fact]
        public void cursor_for()
        {
            var sql = @"cursor for 
		  select Id, name from customer with (nolock)";

            Parse(sql);

            ThenExprShouldBe(@"CURSOR FOR SELECT Id, name
FROM customer WITH( nolock )");
        }

        [Fact]
        public void open()
        {
            var sql = @"open @mydata";

            Parse(sql);

            ThenExprShouldBe(@"OPEN @mydata");
        }

        [Fact]
        public void fetch()
        {
            var sql = @"fetch next from @getData 
into @id, @name";

            Parse(sql);

            ThenExprShouldBe(@"FETCH NEXT FROM @getData
INTO @id, @name");
        }


        [Fact]
        public void between()
        {
            var sql = @"1 between a-1 and b-2";

            Parse(sql);

            ThenExprShouldBe(@"1 BETWEEN a - 1 AND b - 2");
        }
        
        
        [Fact]
        public void constraint()
        {
            var sql = @"CONSTRAINT [PK_my] PRIMARY KEY CLUSTERED ([id] ASC, [name] ASC)";

            Parse(sql);

            ThenExprShouldBe(@"CONSTRAINT [PK_my] PRIMARY KEY CLUSTERED([id] ASC, [name] ASC)");
        }
        
        
        [Fact]
        public void alter_database()
        {
            var sql = @"alter database [$(DatabaseName)] ADD FILEGROUP [file123];";

            Parse(sql);

            ThenExprShouldBe(@"ALTER DATABASE [$(DatabaseName)] ADD FILEGROUP [file123] ;");
        }
        
        
        
        [Fact]
        public void batch_reference_file()
        {
            var sql = @":r file";

            Parse(sql);

            ThenExprShouldBe(@":r file");
        }
        
        
        [Fact]
        public void drop_role()
        {
            var sql = @"drop role roleName";

            Parse(sql);

            ThenExprShouldBe(@"DROP ROLE roleName");
        }
        
        
    }
}