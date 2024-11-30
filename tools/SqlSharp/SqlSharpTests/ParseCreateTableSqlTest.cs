using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.Expressions;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

[TestFixture]
public class ParseCreateTableSqlTest
{
    [Test]
    public void computed_column_definition()
    {
        var sql = $"""
                   CREATE TABLE tb1 (
                       [id] INT,
                       [PartitionHash]  AS ([id]%(10)) PERSISTED NOT NULL
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "INT"
                },
                new SqlComputedColumnDefinition
                {
                    ColumnName = "[PartitionHash]",
                    Expression = "([id]%(10))",
                    IsPersisted = true,
                    IsNotNull = true
                }
            ],
        });
    }
    
    [Test]
    public void DefaultNegativeNumber()
    {
        var sql = $"""
                   CREATE TABLE tb1
                   (
                       [id] INT NOT NULL DEFAULT -1
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "INT",
                    IsNullable = false,
                    Constraints = [
                        new SqlConstraintDefaultValue
                        {
                            DefaultValue = "-1"
                        }
                    ]
                }
            ]
        });
    }
    
    [Test]
    public void SchemaName_dot_TableName()
    {
        var sql = $"""
                   CREATE TABLE dbo.tb1
                   (
                    id int
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SqlCreateTableExpression
        {
            TableName = "dbo.tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "id",
                    DataType = "int"
                }
            ]
        });
    }
    
    [Test]
    public void CreateTableName_Comment_Fields()
    {
        var sql = $"""
                   CREATE TABLE tb1 -- comment
                   (
                       [id] int, /* Identify transfer in OL */     
                       [name] DECIMAL (19,6) NOT NULL
                   );
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "int"
                },
                new SqlColumnDefinition
                {
                    ColumnName = "[name]",
                    DataType = "DECIMAL",
                    DataSize = new SqlDataSize()
                    {
                        Size = "19",
                        Scale = 6
                    },
                    IsNullable = false
                }
            ]
        });
    }
    
    [Test]
    public void DefaultFloat()
    {
        var sql = $"""
                   CREATE TABLE tb1(
                      [id] DECIMAL(18, 2) NOT NULL DEFAULT 0.0 
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "DECIMAL",
                    DataSize = new SqlDataSize()
                    {
                        Size = "18",
                        Scale = 2,
                    },
                    IsNullable = false,
                    Constraints = [
                        new SqlConstraintDefaultValue
                        {
                            DefaultValue = "0.0"
                        }
                    ]
                }
            ]
        });
    }
    
    [Test]
    public void TableForeignKeyReferences()
    {
        var sql = $"""
                   CREATE TABLE tb1 (
                   CONSTRAINT [FK1] FOREIGN KEY ([Id]) REFERENCES tb2 ([id2])
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Constraints = [
                new SqlConstraintForeignKey
                {
                    ConstraintName = "[FK1]",
                    Columns = [
                        new SqlConstraintColumn
                        {
                            ColumnName = "[Id]",
                            Order = ""
                        }
                    ],
                    ReferencedTableName = "tb2",
                    RefColumn = "[id2]"
                }
            ]
        });
    }
    
    [Test]
    public void SqlIdentifierName()
    {
        var sql = $"""
                   CREATE TABLE [dbo].db1 (
                       [id] int 
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "[dbo].db1",
            Columns = [
                new SqlColumnDefinition
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "NVARCHAR",
                    DataSize = new SqlDataSize
                    {
                        Size = "50",
                    },
                },
                new SqlColumnDefinition
                {
                    ColumnName = "[name]",
                    DataType = "varchar",
                    DataSize = new SqlDataSize
                    {
                        Size = "10",
                    },
                    Constraints = [
                        new SqlConstraintDefaultValue
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "VARCHAR",
                    DataSize = new SqlDataSize
                    {
                        Size = "30",
                    },
                    IsNullable = false
                },
                new SqlColumnDefinition
                {
                    ColumnName = "[Id2]",
                    DataType = "VARCHAR",
                    DataSize = new SqlDataSize
                    {
                        Size = "50",
                    },
                    IsNullable = true,
                    Constraints = [
                        new SqlConstraintDefaultValue
                        {
                            ConstraintName = "[D1]",
                            DefaultValue = "N'0'"
                        }
                    ]
                }
            ],
            Constraints = [
                new SqlConstraintPrimaryKeyOrUnique
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "#tmp1",
            Columns =
            [
                new SqlColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "int",
                    IsNullable = false
                },
                new SqlColumnDefinition
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "INT",
                    IsNullable = false,
                    Constraints = [
                        new SqlConstraintPrimaryKeyOrUnique
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
                new SqlColumnDefinition
                {
                    ColumnName = "[name]",
                    DataType = "NVARCHAR",
                    DataSize = new SqlDataSize
                        {
                    Size = "50",
                        },
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "[dbo].[VipBetSetting]",
            Columns =
            [
                new SqlColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "INT",
                    IsNullable = false,
                    Identity = new SqlIdentity{ Seed = 1, Increment = 1 },
                },
                new SqlColumnDefinition
                {
                    ColumnName = "[CreatedOn]",
                    DataType = "DATETIME",
                    IsNullable = false,
                    Constraints =
                    [
                        new SqlConstraintDefaultValue
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
        rc.ShouldBe(new SqlCreateTableExpression
        {
            TableName = "#tb",
            Columns = [
                new SqlColumnDefinition
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "$tmp",
            Columns =
            [
                new SqlColumnDefinition
                {
                    ColumnName = "[IsUat]",
                    DataType = "BIT",
                    IsNullable = false
                }
            ],
            Constraints =
            [
                new SqlConstraintPrimaryKeyOrUnique
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
        rc.ShouldBe(new SqlCreateTableExpression
        {
            TableName = "Persons",
            Columns =
            [
                new SqlColumnDefinition { ColumnName = "id", DataType = "int" },
                new SqlColumnDefinition
                {
                    ColumnName = "[Id]",
                    IsPrimaryKey = true,
                    DataType = "BIGINT",
                    Identity = new SqlIdentity { Seed = 1, Increment = 1 },
                    IsNullable = false,
                    NotForReplication = false,
                    Constraints = [],
                },
                new SqlColumnDefinition { ColumnName = "LastName", DataType = "varchar", DataSize = new SqlDataSize{ Size = "50"}, },
                new SqlColumnDefinition { ColumnName = "Money", DataType = "decimal", DataSize = new SqlDataSize(){Size = "10", Scale = 3} },
                new SqlColumnDefinition
                {
                    ColumnName = "[name]", DataType = "[int]",
                    Identity = new SqlIdentity
                    {
                        Seed = 1,
                        Increment = 1,
                    },
                },
                new SqlColumnDefinition
                {
                    ColumnName = "[name2]", DataType = "[int]",
                    Identity = new SqlIdentity
                    {
                        Seed = 1,
                        Increment = 1,
                    },
                },
                new SqlColumnDefinition
                {
                    ColumnName = "cname", DataType = "[int]",
                    IsNullable = true,
                },
                new SqlColumnDefinition
                {
                    ColumnName = "[rid]", DataType = "[int]",
                    Identity = new SqlIdentity()
                    {
                        Seed = 1,
                        Increment = 1,
                    },
                    NotForReplication = true,
                },
                new SqlColumnDefinition
                {
                    ColumnName = "[DailyTotalRaw]", DataType = "DECIMAL",
                    DataSize = new SqlDataSize(){
                    Size = "19",
                    Scale = 6,},
                    Constraints =
                    [
                        new SqlConstraintDefaultValue
                        {
                            ConstraintName = "[DF_CheckSum]",
                            DefaultValue = "(0)",
                        }
                    ]
                }
            ],
            Constraints =
            [
                new SqlConstraintPrimaryKeyOrUnique
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "VARCHAR",
                    DataSize = new  SqlDataSize(){Size = "3"},
                    IsNullable = false,
                    Constraints = [
                        new SqlConstraintDefaultValue
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[day]",
                    DataType = "DATETIME",
                    IsNullable = false,
                    Constraints = [
                        new SqlConstraintDefaultValue
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "[Banner]",
            Columns =
            [
                new SqlColumnDefinition
                {
                    ColumnName = "[BannerType]",
                    DataType = "INT",
                    Constraints = [
                        new SqlConstraintDefaultValue
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "#tmp",
            Columns =
            [
                new SqlColumnDefinition
                {
                    ColumnName = "AuditCount",
                    DataType = "nvarchar",
                    DataSize = new SqlDataSize{Size = "MAX"},
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "INT",
                    IsNullable = false,
                    Identity = new SqlIdentity { Seed = 1, Increment = 1 },
                    IsPrimaryKey = true
                },
                new SqlColumnDefinition
                {
                    ColumnName = "[LoginName]",
                    DataType = "NVARCHAR",
                    DataSize = new SqlDataSize(){ Size = "50" },
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "int",
                }
            ],
            Constraints = [
                new SqlConstraintPrimaryKeyOrUnique
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "#CustIdList",
            Columns =
            [
                new SqlColumnDefinition
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
        rc.ShouldBe(new SqlCreateTableExpression
        {
            TableName = "#tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "INT",
                }
            ],
            Constraints = [
                new SqlConstraintPrimaryKeyOrUnique
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
                new SqlConstraintPrimaryKeyOrUnique
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
        rc.ShouldBe(new SqlCreateTableExpression
        {
            TableName = "#tmp",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[id]",
                    DataType = "NVARCHAR",
                    DataSize = new SqlDataSize(){Size = "50"},
                    IsNullable = true
                }
            ],
            Constraints = [
                new SqlConstraintPrimaryKeyOrUnique
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
        rc.ShouldBe(new SqlCreateTableExpression
        {
            TableName = "[dbo].[CashSettled]",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[custid]",
                    DataType = "INT",
                    IsNullable = false
                }
            ],
            Constraints = [
                new SqlConstraintPrimaryKeyOrUnique
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
        rc.ShouldBe(new SqlCreateTableExpression
        {
            TableName = "#tb1",
            Columns = [
                new SqlColumnDefinition
                {
                    ColumnName = "[Id]",
                    DataType = "int",
                }
            ],
            Constraints = [
                new SqlConstraintPrimaryKeyOrUnique
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
        rc.ShouldBe(new SqlCreateTableExpression()
        {
            TableName = "[dbo].[UserTracking]",
            Columns = [
                new SqlColumnDefinition
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
                new SqlColumnDefinition
                {
                    ColumnName = "[Extra]",
                    DataType = "nvarchar",
                    DataSize = new SqlDataSize(){Size = "4000"},
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