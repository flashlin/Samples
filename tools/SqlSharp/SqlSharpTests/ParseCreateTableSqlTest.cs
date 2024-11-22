using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

[TestFixture]
public class ParseCreateTableSqlTest
{
    [Test]
    public void SqlIdentifierName()
    {
        var sql = $"""
                   CREATE TABLE [dbo].db1 (
                       [id] int 
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "[dbo].db1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "int"
                }
            ]
        });
    }
    
    [Test]
    public void LastColumnAllowComma()
    {
        var sql = $"""
                   CREATE TABLE tb1(
                       [id] NVARCHAR(50),
                       [name] varchar(10) Default GetDate(),  
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "tb1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "NVARCHAR",
                    Size = "50"
                },
                new ColumnDefinition
                {
                    ColumnName = "[name]",
                    DataType = "varchar",
                    Size = "10",
                    Constraints = [
                        new SqlConstraint
                        {
                            DefaultValue = "GetDate()"
                        }
                    ]
                }
            ]
        });
    }
    
    [Test]
    public void Column_Unique_Column()
    {
        var sql = $"""
                   CREATE TABLE tb1(
                   [Id]    VARCHAR (30) NOT NULL, 
                    CONSTRAINT [U1] UNIQUE(ExternalRefNo) ,
                    [Id2]  VARCHAR(50)  NULL CONSTRAINT [D1]  DEFAULT (N'0')
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "tb1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "VARCHAR",
                    Size = "30",
                    IsNullable = false
                },
                new ColumnDefinition
                {
                    ColumnName = "[Id2]",
                    DataType = "VARCHAR",
                    Size = "50",
                    IsNullable = true,
                    Constraints = [
                        new SqlConstraint
                        {
                            ConstraintName = "[D1]",
                            DefaultValue = "N'0'"
                        }
                    ]
                }
            ],
            Constraints = [
                new SqlConstraint
                {
                    ConstraintName = "[U1]",
                    ConstraintType = "UNIQUE",
                    Clustered = "",
                    Columns = [
                        new SqlConstraintColumn
                        {
                            ColumnName = "ExternalRefNo",
                            Order = ""
                        }
                    ]
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
    public void ColumnConstraintWithIdentity()
    {
        var sql = $"""
                   CREATE TABLE tb1
                   (
                   	[Id] INT NOT NULL CONSTRAINT [pk1] PRIMARY KEY NONCLUSTERED ([Id] ASC) WITH (FILLFACTOR = 90) IDENTITY, 
                    [name] NVARCHAR(50) NULL 
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "tb1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "INT",
                    IsNullable = false,
                    Constraints = [
                        new SqlConstraint
                        {
                            ConstraintName = "[pk1]",
                            ConstraintType = "PRIMARY KEY",
                            Clustered = "NONCLUSTERED",
                            Columns = [
                                new SqlConstraintColumn
                                {
                                    ColumnName = "[Id]",
                                    Order = "ASC"
                                }
                            ],
                            WithToggles = [
                                new SqlToggle
                                {
                                    ToggleName = "FILLFACTOR",
                                    Value = "90"
                                }
                            ],
                            Identity = new SqlIdentity()
                        }
                    ]
                },
                new ColumnDefinition
                {
                    ColumnName = "[name]",
                    DataType = "NVARCHAR",
                    Size = "50",
                    IsNullable = true
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
                        new SqlConstraint
                        {
                            DefaultValue = "GetDate()"
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
                        new SqlConstraint
                        {
                            ConstraintName = "[DF_CheckSum]",
                            DefaultValue = "(0)",
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
                        new SqlToggle
                        {
                            ToggleName = "PAD_INDEX",
                            Value = "OFF"
                        },
                        new SqlToggle
                        {
                            ToggleName = "STATISTICS_NORECOMPUTE",
                            Value = "OFF"
                        },
                        new SqlToggle
                        {
                            ToggleName = "IGNORE_DUP_KEY",
                            Value = "OFF"
                        },
                        new SqlToggle
                        {
                            ToggleName = "ALLOW_ROW_LOCKS",
                            Value = "ON"
                        },
                        new SqlToggle
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
    public void Default_EmptyString()
    {
        var sql = $"""
                   CREATE TABLE tb1
                   (
                   	[id] VARCHAR(3) NOT NULL CONSTRAINT [DF_1] DEFAULT ''
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "tb1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "VARCHAR",
                    Size = "3",
                    IsNullable = false,
                    Constraints = [
                        new SqlConstraint
                        {
                            ConstraintName = "[DF_1]",
                            DefaultValue = "''"
                        }
                    ]
                }
            ]
        });
    }

    [Test]
    public void DefaultDateWithoutQuoted
        ()
    {
        var sql = $"""
                   CREATE TABLE tb1
                   (
                       [day] DATETIME NOT NULL DEFAULT 2019-01-01 
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "tb1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[day]",
                    DataType = "DATETIME",
                    IsNullable = false,
                    Constraints = [
                        new SqlConstraint
                        {
                            DefaultValue = "2019-01-01"
                        }
                    ]
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
                        new SqlConstraint
                        {
                            ConstraintName = "DEFAULT",
                            DefaultValue = "NULL"
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
    public void FieldNotNull_Identity_PrimaryKey()
    {
        var sql = $"""
                   CREATE TABLE tb1 (
                   	[Id]		INT				NOT NULL IDENTITY(1,1) PRIMARY KEY,
                   	[LoginName]	NVARCHAR (50)   NULL
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "tb1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "INT",
                    IsNullable = false,
                    Identity = new SqlIdentity { Seed = 1, Increment = 1 },
                    IsPrimaryKey = true
                },
                new ColumnDefinition
                {
                    ColumnName = "[LoginName]",
                    DataType = "NVARCHAR",
                    Size = "50",
                    IsNullable = true
                }
            ]
        });
    }

    [Test]
    public void LastFieldNoCommaAndTableConstraint()
    {
        var sql = $"""
                   CREATE TABLE tb1(
                   	[Id] int 
                   PRIMARY KEY CLUSTERED 
                   (
                   	[Id] ASC
                   )WITH (PAD_INDEX = OFF) ON [PRIMARY]
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement()
        {
            TableName = "tb1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "int",
                }
            ],
            Constraints = [
                new SqlConstraint
                {
                    ConstraintName = "",
                    ConstraintType = "PRIMARY KEY",
                    Clustered = "CLUSTERED",
                    Columns = [
                        new SqlConstraintColumn
                        {
                            ColumnName = "[Id]",
                            Order = "ASC" 
                        }
                    ],
                    WithToggles = [
                        new SqlToggle
                        {
                            ToggleName = "PAD_INDEX",
                            Value = "OFF"
                        }
                    ],
                    On = "[PRIMARY]"
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
    public void MultipleTableConstraints()
    {
        var sql = $"""
                   CREATE TABLE #tb1 (
                       [id] INT,
                       CONSTRAINT [PK_1] PRIMARY KEY CLUSTERED ([id] ASC),
                       CONSTRAINT [UQ_1] UNIQUE NONCLUSTERED ([name] ASC)
                   );
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement
        {
            TableName = "#tb1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "INT",
                }
            ],
            Constraints = [
                new SqlConstraint
                {
                    ConstraintName = "[PK_1]",
                    ConstraintType = "PRIMARY KEY",
                    Clustered = "CLUSTERED",
                    Columns = [
                        new SqlConstraintColumn
                        {
                            ColumnName = "[id]",
                            Order = "ASC"
                        }
                    ],
                },
                new SqlConstraint
                {
                    ConstraintName = "[UQ_1]",
                    ConstraintType = "UNIQUE",
                    Clustered = "NONCLUSTERED",
                    Columns = [
                        new SqlConstraintColumn
                        {
                            ColumnName = "[name]",
                            Order = "ASC"
                        }
                    ],
                }
            ] 
        });
    }

    [Test]
    public void TableConstraintUniqueNonClustered()
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
                        new SqlToggle
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
    public void TablePrimaryKeyWithOnPrimary()
    {
        var sql = $"""
                   CREATE TABLE #tb1(
                   	[Id] int,
                   PRIMARY KEY CLUSTERED 
                   (
                   	[Id] ASC
                   )WITH (PAD_INDEX = OFF) ON [PRIMARY]
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new CreateTableStatement
        {
            TableName = "#tb1",
            Columns = [
                new ColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "int",
                }
            ],
            Constraints = [
                new SqlConstraint
                {
                    ConstraintName = "",
                    ConstraintType = "PRIMARY KEY",
                    Clustered = "CLUSTERED",
                    Columns = [
                        new SqlConstraintColumn
                        {
                            ColumnName = "[Id]",
                            Order = "ASC" 
                        }
                    ],
                    WithToggles = [
                        new SqlToggle
                        {
                            ToggleName = "PAD_INDEX",
                            Value = "OFF"
                        }
                    ],
                    On = "[PRIMARY]"
                }
            ]
        });
    }

    [Test]
    public void WithoutTableConstraint()
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

    private ParseResult<ISqlExpression> ParseSql(string sql)
    {
        return SqlParser.Parse(sql);
    }
}