using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

[TestFixture]
public class ParseCreateTableSqlTest
{
    [Test]
    public void AddExtendedProperty()
    {
        var sql = $"""
                   EXEC sp_addextendedproperty
                   @name = N'MS_Description',        -- 屬性名稱（固定為 MS_Description 用於說明）
                   @value = N'hello',                -- 說明內容 
                   @level0type = N'SCHEMA',          -- 第 1 級目標類型
                   @level0name = N'dbo',             -- 第 1 級名稱（預設 schema）
                   @level1type = N'TABLE',           -- 第 2 級目標類型
                   @level1name = N'customer',        -- 第 2 級名稱（資料表名稱）
                   @level2type = N'COLUMN',          -- 第 3 級目標類型
                   @level2name = 'addr';            -- 第 3 級名稱（欄位名稱)
                   """;

        var rc = ParseSql(sql);

        rc.ShouldBe(new SqlSpAddExtendedProperty()
        {
            Name = "N'MS_Description'",
            Value = "N'hello'",
            Level0Type = "N'SCHEMA'",
            Level0Name = "N'dbo'",
            Level1Type = "N'TABLE'",
            Level1Name = "N'customer'",
            Level2Type = "N'COLUMN'",
            Level2Name = "'addr'",
        });
    }

    [Test]
    public void ConstraintUnique()
    {
        var sql = $"""
                   CREATE TABLE $tmp
                   (
                       [IsUat] BIT NOT NULL,
                       CONSTRAINT UC_1 UNIQUE (KeyName,Lang)
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "$tmp",
            Columns =
            [
                new ColumnDefinition
                {
                    ColumnName = "[IsUat]",
                    DataType = "BIT",
                    IsNullable = false
                }
            ],
            Constraints =
            [
                new SqlConstraint
                {
                    ConstraintName = "UC_1",
                    ConstraintType = "UNIQUE",
                    Clustered = "",
                    Columns =
                    [
                        new SqlConstraintColumn
                        {
                            ColumnName = "KeyName",
                            Order = ""
                        },
                        new SqlConstraintColumn
                        {
                            ColumnName = "Lang",
                            Order = ""
                        }
                    ]
                }
            ]
        });
    }

    [Test]
    public void DefinitionDataMax()
    {
        var sql =$"""
                  CREATE TABLE #tmp(
                  	 AuditCount nvarchar(max)
                  ) 
                  """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "#tmp",
            Columns =
            [
                new ColumnDefinition
                {
                    ColumnName = "AuditCount",
                    DataType = "nvarchar",
                    Size = "MAX"
                }
            ]
        });
    }

    [Test]
    public void ColumnCommentColumn()
    {
        var sql = $"""
                  create table #tmp1
                  (              
                    [id] int NOT NULL,  
                    -- comment --  
                    uid int
                  )               
                  """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "#tmp1",
            Columns =
            [
                new ColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "int",
                    IsNullable = false
                },
                new ColumnDefinition
                {
                    ColumnName = "uid",
                    DataType = "int"
                }
            ]
        });
    }

    [Test]
    public void LowerCaseCreateTable()
    {
        var sql = $"""
                   create table #CustIdList (  
                    CustID int
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "#CustIdList",
            Columns =
            [
                new ColumnDefinition
                {
                    ColumnName = "CustID",
                    DataType = "int"
                }
            ]
        });
    }

    [Test]
    public void ColumnDefaultValueWithoutConstraint()
    {
        var sql = $"""
                   CREATE TABLE [dbo].[VipBetSetting]
                   (
                   	[Id] INT NOT NULL IDENTITY, 
                   	[CreatedOn] DATETIME NOT NULL DEFAULT GetDate()
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "[dbo].[VipBetSetting]",
            Columns =
            [
                new ColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "INT",
                    IsNullable = false,
                    Identity = new SqlIdentity{ Seed = 1, Increment = 1 },
                },
                new ColumnDefinition
                {
                    ColumnName = "[CreatedOn]",
                    DataType = "DATETIME",
                    IsNullable = false,
                    Constraints =
                    [
                        new SqlConstraintDefault
                        {
                            ConstraintName = "DEFAULT",
                            Value = "GetDate()"
                        }
                    ]
                }
            ]
        });
    }

    [Test]
    public void CreateTable()
    {
        var sql = $"""
                   CREATE TABLE Persons (
                   id int,
                   [Id] BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
                   LastName varchar(50),
                   Money decimal(10,3),
                   [name] [int] IDENTITY(1,1) NOT NULL,
                   [name2] [int] NOT NULL IDENTITY(1,1),
                   cname [int] NULL,
                   [rid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL, 
                   [DailyTotalRaw]  DECIMAL (19, 6) CONSTRAINT [DF_CheckSum] DEFAULT ((0)) NOT NULL,
                   CONSTRAINT [PK_AcceptedBets] PRIMARY KEY CLUSTERED 
                     (
                   	    [MatchResultID] ASC
                     ) WITH (PAD_INDEX  = OFF, STATISTICS_NORECOMPUTE  = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS  = ON, ALLOW_PAGE_LOCKS  = ON) ON [PRIMARY]
                   );
                   """;

        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement
        {
            TableName = "Persons",
            Columns =
            [
                new ColumnDefinition { ColumnName = "id", DataType = "int" },
                new ColumnDefinition
                {
                    ColumnName = "[Id]",
                    IsPrimaryKey = true,
                    DataType = "BIGINT",
                    Identity = new SqlIdentity { Seed = 1, Increment = 1 },
                    IsNullable = false,
                    NotForReplication = false,
                    Constraints = [],
                },
                new ColumnDefinition { ColumnName = "LastName", DataType = "varchar", Size = "50" },
                new ColumnDefinition { ColumnName = "Money", DataType = "decimal", Size = "10", Scale = 3 },
                new ColumnDefinition
                {
                    ColumnName = "[name]", DataType = "[int]",
                    Identity = new SqlIdentity
                    {
                        Seed = 1,
                        Increment = 1,
                    },
                },
                new ColumnDefinition
                {
                    ColumnName = "[name2]", DataType = "[int]",
                    Identity = new SqlIdentity
                    {
                        Seed = 1,
                        Increment = 1,
                    },
                },
                new ColumnDefinition
                {
                    ColumnName = "cname", DataType = "[int]",
                    IsNullable = true,
                },
                new ColumnDefinition
                {
                    ColumnName = "[rid]", DataType = "[int]",
                    Identity = new SqlIdentity()
                    {
                        Seed = 1,
                        Increment = 1,
                    },
                    NotForReplication = true,
                },
                new ColumnDefinition
                {
                    ColumnName = "[DailyTotalRaw]", DataType = "DECIMAL",
                    Size = "19",
                    Scale = 6,
                    Constraints =
                    [
                        new SqlConstraintDefault
                        {
                            ConstraintName = "[DF_CheckSum]",
                            Value = "(0)",
                        }
                    ]
                }
            ],
            Constraints =
            [
                new SqlConstraint
                {
                    ConstraintName = "[PK_AcceptedBets]",
                    ConstraintType = "PRIMARY KEY",
                    Clustered = "CLUSTERED",
                    Columns =
                    [
                        new SqlConstraintColumn
                        {
                            ColumnName = "[MatchResultID]",
                            Order = "ASC"
                        }
                    ],
                    WithToggles =
                    [
                        new SqlWithToggle
                        {
                            ToggleName = "PAD_INDEX",
                            Value = "OFF"
                        },
                        new SqlWithToggle
                        {
                            ToggleName = "STATISTICS_NORECOMPUTE",
                            Value = "OFF"
                        },
                        new SqlWithToggle
                        {
                            ToggleName = "IGNORE_DUP_KEY",
                            Value = "OFF"
                        },
                        new SqlWithToggle
                        {
                            ToggleName = "ALLOW_ROW_LOCKS",
                            Value = "ON"
                        },
                        new SqlWithToggle
                        {
                            ToggleName = "ALLOW_PAGE_LOCKS",
                            Value = "ON"
                        }
                    ],
                    On = "[PRIMARY]"
                }
            ]
        });
    }

    [Test]
    public void DefaultNull()
    {
        var sql = $"""
                   CREATE TABLE [Banner]
                   (
                   [BannerType] INT DEFAULT NULL 
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "[Banner]",
            Columns =
            [
                new ColumnDefinition
                {
                    ColumnName = "[BannerType]",
                    DataType = "INT",
                    Constraints = [
                        new SqlConstraintDefault
                        {
                            ConstraintName = "DEFAULT",
                            Value = "NULL"
                        }
                    ]
                }
            ]
        });
    }

    [Test]
    public void TableConstraintWithoutOn()
    {
        var sql = $"""
                   CREATE TABLE [dbo].[CashSettled] (
                       [custid] INT NOT NULL,
                       CONSTRAINT [PK_CashSettled] PRIMARY KEY CLUSTERED ([custid] ASC) WITH (FILLFACTOR = 85)
                   );
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement
        {
            TableName = "[dbo].[CashSettled]",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[custid]",
                    DataType = "INT",
                    IsNullable = false
                }
            ],
            Constraints = [
                new SqlConstraint
                {
                    ConstraintName = "[PK_CashSettled]",
                    ConstraintType = "PRIMARY KEY",
                    Clustered = "CLUSTERED",
                    Columns = [
                        new SqlConstraintColumn
                        {
                            ColumnName = "[custid]",
                            Order = "ASC"
                        }
                    ],
                    WithToggles = [
                        new SqlWithToggle
                        {
                            ToggleName = "FILLFACTOR",
                            Value = "85"
                        }
                    ]
                }
            ]
        });
    }

    [Test]
    public void ColumnName1Len()
    {
        var sql = $"""
                   Create table #tb  (        
                     R float  
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement
        {
            TableName = "#tb",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "R",
                    DataType = "float",
                }
            ],
        });
    }

    [Test]
    public void UniqueNonClustered()
    {
        var sql = $"""
                   CREATE TABLE #tmp
                   (
                       [id] NVARCHAR (50) NULL,
                       CONSTRAINT [UQ_1] UNIQUE NONCLUSTERED ([ID] ASC, [Project] ASC)
                   );
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement
        {
            TableName = "#tmp",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "NVARCHAR",
                    Size = "50",
                    IsNullable = true
                }
            ],
            Constraints = [
                new SqlConstraint
                {
                    ConstraintName = "[UQ_1]",
                    ConstraintType = "UNIQUE",
                    Clustered = "NONCLUSTERED",
                    Columns = [
                        new SqlConstraintColumn
                        {
                            ColumnName = "[ID]",
                            Order = "ASC"
                        },
                        new SqlConstraintColumn
                        {
                            ColumnName = "[Project]",
                            Order = "ASC"
                        }
                    ]
                }
            ]
        });
    }

    [Test]
    public void WithoutConstraint()
    {
        var sql = $"""
                   CREATE TABLE [dbo].[UserTracking]
                   (
                   	[Id] BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
                   	[Extra] nvarchar(4000) NULL
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "[dbo].[UserTracking]",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "BIGINT",
                    Identity = new SqlIdentity
                    {
                        Seed = 1,
                        Increment = 1
                    },
                    IsPrimaryKey = true,
                    IsNullable = false
                },
                new ColumnDefinition
                {
                    ColumnName = "[Extra]",
                    DataType = "nvarchar",
                    Size = "4000",
                    IsNullable = true
                }
            ]
        });
    }

    private static Either<ISqlExpression, ParseError> ParseSql(string sql)
    {
        var p = new SqlParser(sql);
        var rc = p.Parse();
        return rc;
    }
}