using T1.CodeDom.Core;
using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateTableTest : TestBase
{
    public CreateTableTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
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
name VARCHAR (10)
)");
    }

    [Fact]
    public void create_table_not_for_replication()
    {
        var sql = @"CREATE TABLE [dbo].[customer] (
    [id] int NOT FOR REPLICATION NOT NULL
);";

        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [id] INT NOT FOR REPLICATION NOT NULL
) ;");
    }


    [Fact]
    public void create_table_primary_key_clustered()
    {
        var sql = @"create table [dbo].[customer] (
    [id]      INT            NOT NULL,
    PRIMARY KEY CLUSTERED ([id] ASC)
);";

        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [id] INT NOT NULL,
    PRIMARY KEY CLUSTERED([id] ASC)
) ;");
    }


    [Fact]
    public void create_table_without_comma_clustered()
    {
        var sql = @"create table [dbo].[customer] (
    [id]      INT            NOT NULL
    PRIMARY KEY CLUSTERED ([id] ASC)
);";

        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [id] INT NOT NULL
    PRIMARY KEY CLUSTERED([id] ASC)
) ;");
    }


    [Fact]
    public void create_table_with_toggle()
    {
        var sql = @"create table [dbo].[customer] (
    [id]      INT            NOT NULL
PRIMARY KEY CLUSTERED ( [Id] ASC )
WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) 
ON [PRIMARY]
);";

        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [id] INT NOT NULL
    PRIMARY KEY CLUSTERED([Id] ASC)
    WITH(PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON)
    ON [PRIMARY]
) ;");
    }


    [Fact]
    public void create_table_on_primary()
    {
        var sql = @"create table [dbo].[customer] (
    [id]      INT            NOT NULL
) ON [PRIMARY] ;";
        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [id] INT NOT NULL )
    ON [PRIMARY] ;");
    }


    [Fact]
    public void create_table_comma_with()
    {
        var sql = @"CREATE TABLE [dbo].[customer] (
	[Id] INT IDENTITY(1, 1) NOT NULL,
	[sid] INT NULL,
	PRIMARY KEY CLUSTERED ([Id] ASC) WITH (
		PAD_INDEX = OFF
		) ON [PRIMARY]
	) ON [PRIMARY]";
        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [Id] INT IDENTITY (1,1) NOT NULL,
    [sid] INT NULL,
    PRIMARY KEY CLUSTERED([Id] ASC)
    WITH(PAD_INDEX = OFF)
    ON [PRIMARY]
    )
    ON [PRIMARY]");
    }


    [Fact]
    public void create_table_primary_key_columns()
    {
        var sql = @"CREATE TABLE [dbo].[customer] (
	[Id] INT IDENTITY(1, 1) NOT NULL,
	[sid] INT NULL,
	CONSTRAINT PK_customer PRIMARY KEY ([Id])
	)";
        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [Id] INT IDENTITY (1,1) NOT NULL,
    [sid] INT NULL,
CONSTRAINT PK_customer PRIMARY KEY ([Id])
)");
    }


    [Fact]
    public void create_table_default_getdate()
    {
        var sql = @"CREATE TABLE [dbo].[customer] (
	[Id] INT,
	[birth] datetime DEFAULT GETDATE()
	)";
        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer]( [Id] INT, [birth] DATETIME DEFAULT GETDATE() )");
    }


    [Fact]
    public void create_table_constraint_constraint()
    {
        var sql = @"CREATE TABLE [dbo].[customer] (
	[Id] INT,
    IsUpdated bit NOT NULL CONSTRAINT [DF_IsUpdated] DEFAULT ((0))
    CONSTRAINT [pk_customer] PRIMARY KEY CLUSTERED ([id] ASC)
	)";

        Parse(sql);
        
        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer]( 
    [Id] INT, 
    IsUpdated BIT NOT NULL CONSTRAINT [DF_IsUpdated] DEFAULT ( 0 ),
    CONSTRAINT [pk_customer] PRIMARY KEY CLUSTERED([id] ASC)
    )");
    }


    [Fact]
    public void create_temp_table()
    {
        var sql = @"CREATE TABLE #tmpCustomer (id int, DATE DATETIME)";
        Parse(sql);
        ThenExprShouldBe(@"CREATE TABLE #tmpCustomer( id INT, DATE DATETIME )");
    }
    
    [Fact]
    public void create_table_unique()
    {
        var sql = @"CREATE TABLE customer ( id int, CONSTRAINT [ct_customer] UNIQUE ([id]) )";
        Parse(sql);
        ThenExprShouldBe(@"CREATE TABLE customer( id INT, CONSTRAINT [ct_customer] UNIQUE([id] ASC) )");
    }
    
    [Fact]
    public void create_table_default_date()
    {
        var sql = @"CREATE TABLE customer ( birth datetime default 2022-01-01 )";
        Parse(sql);
        ThenExprShouldBe(@"CREATE TABLE customer( birth DATETIME DEFAULT 2022-01-01 )");
    }
    
}