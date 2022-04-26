using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
    public class CreateTest : TestBase
    {
        public CreateTest(ITestOutputHelper outputHelper) : base(outputHelper)
        {
        }

        [Fact]
        public void create_procedure()
        {
            var sql = @"create procedure myProc
@id int,
@name varchar(50)
as
begin
	set noexec on;
end";
            Parse(sql);

            ThenExprShouldBe(@"CREATE PROCEDURE myProc
@id INT, 
@name VARCHAR(50)
AS
BEGIN
	SET NOEXEC ON
	; 
END");
        }

        [Fact]
        public void create_procedure_arg1_eq()
        {
            var sql = @"create procedure myProc
@id int,
@name varchar(50) = 'a'
as
begin
	set noexec on;
end";
            Parse(sql);

            ThenExprShouldBe(@"CREATE PROCEDURE myProc
@id INT, 
@name VARCHAR(50) = 'a'
AS
BEGIN
	SET NOEXEC ON
	; 
END");
        }


        [Fact]
        public void create_table()
        {
            var sql = @"create table #cust (  
        ID int,
		birth datetime
    )";
            Parse(sql);

            ThenExprShouldBe(@"CREATE TABLE #cust(
ID INT,
birth DATETIME
)");
        }

        [Fact]
        public void create_table_comment()
        {
            var sql = @"create table #customer (       
id int,
   -- test --    
name varchar(10)
  )";
            Parse(sql);

            ThenExprShouldBe(@"CREATE TABLE #customer(
id INT,
name VARCHAR(10)
)");
        }

        [Fact]
        public void create_clustered_index()
        {
            var sql = @"create clustered index ix_id on #customer (id)";
            Parse(sql);

            ThenExprShouldBe(@"CREATE CLUSTERED INDEX ix_id ON #customer(id)");
        }

        [Fact]
        public void create_tmpTable_not_null()
        {
            var sql = @"create table #tmpCustomer
     (       
			id int NOT NULL
)";
            Parse(sql);

            ThenExprShouldBe(@"CREATE TABLE #tmpCustomer(
id INT NOT NULL
)");
        }

        [Fact]
        public void create_procedure_with_execute_as()
        {
            var sql = @"CREATE PROCEDURE [dbo].[my_proc]
WITH EXECUTE AS 'userName'  
AS
BEGIN
    SET NOCOUNT ON;
END";
            Parse(sql);

            ThenExprShouldBe(@"CREATE PROCEDURE [dbo].[my_proc]
WITH EXECUTE AS 'userName'
AS
BEGIN
    SET NOCOUNT ON ;
END");
        }
        
        
        [Fact]
        public void create_synonym()
        {
            var sql = @"CREATE SYNONYM [dbo].[mySynonym] FOR [RemoteServer].[MyDb].[dbo].[MyTable]";
            
            Parse(sql);

            ThenExprShouldBe(@"CREATE SYNONYM [dbo].[mySynonym] FOR [RemoteServer].[MyDb].[dbo].[MyTable]");
        }
        
        
        [Fact]
        public void create_table_objectId()
        {
            var sql = @"CREATE TABLE [dbo].[myCustomer] (
    [ID]     INT           IDENTITY (1, 1) NOT NULL,
    [Name]   VARCHAR (50)  NOT NULL,
    [Birth] SMALLDATETIME CONSTRAINT [DF_Birth] DEFAULT (getdate()) NOT NULL
)";
            
            Parse(sql);

            ThenExprShouldBe(@"CREATE TABLE [dbo].[myCustomer](
[ID] INT IDENTITY(1,1) NOT NULL,    
[Name] VARCHAR(50) NOT NULL,    
[Birth] SMALLDATETIME CONSTRAINT [DF_Birth] DEFAULT GETDATE() NOT NULL    
)");
        }
        
        
        [Fact]
        public void create_table_constraint_primary_key()
        {
            var sql = @"CREATE TABLE [dbo].[customer] (
    [id]   INT       NOT NULL,
    CONSTRAINT [PK_Id] PRIMARY KEY CLUSTERED ([Id] ASC)
)";
            
            Parse(sql);

            ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [id] INT NOT NULL,
    CONSTRAINT [PK_Id] PRIMARY KEY CLUSTERED([Id] ASC)
)");
        }
        
        
        [Fact]
        public void create_table_with_fillfactor()
        {
            var sql = @"CREATE TABLE [dbo].[customer] (
    [id]   INT       NOT NULL,
    CONSTRAINT [PK_Id] PRIMARY KEY CLUSTERED ([Id] ASC) with (fillfactor = 90)
)";
            
            Parse(sql);

            ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [id] INT NOT NULL,
    CONSTRAINT [PK_Id] PRIMARY KEY CLUSTERED([Id] ASC) WITH(FILLFACTOR = 90)
)");
        }
        
        
        
        [Fact]
        public void create_nonclustered_index_fillfactor()
        {
            var sql = @"CREATE NONCLUSTERED INDEX [ix_customer]
    ON [dbo].[customer]([id] ASC) WITH (FILLFACTOR = 90);";
            
            Parse(sql);

            ThenExprShouldBe(@"CREATE NONCLUSTERED INDEX [ix_customer]
ON [dbo].[customer]([id] ASC) WITH(FILLFACTOR = 90) ;");
        }
    }
}