using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateProcedureTest : TestBase
{
    public CreateProcedureTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void create_procedure_no_parameters()
    {
        var sql = @"create procedure myProc
as
begin
	EXEC [db].[dbo].[fn_my]
end
go";
        Parse(sql);

        ThenExprShouldBe(@"CREATE PROCEDURE myProc
AS
BEGIN
	EXEC [db].[dbo].[fn_my]
END
GO");
    }
    
    [Fact]
    public void create_procedure_cursor_for()
    {
        var sql = @"CREATE PROCEDURE [dbo].[sp_test]
    @id varchar(50) = null -- test1
with execute as owner -- test2
AS
BEGIN

    declare curTable cursor local for
    select id, name from @customer

    open curTable
    fetch next from curTable into @id, @name

    close curTable
    deallocate curTable

END";
        Parse(sql);

        ThenExprShouldBe(@"CREATE PROCEDURE [dbo].[sp_test] 
@id VARCHAR (50) = NULL
WITH EXECUTE AS owner
AS
BEGIN
    DECLARE curTable CURSOR LOCAL FOR SELECT id, name FROM @customer
    OPEN curTable 
    FETCH NEXT FROM curTable INTO @id, @name
    CLOSE curTable
    DEALLOCATE curTable
END");
    }
    
}