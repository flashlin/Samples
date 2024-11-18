using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

[TestFixture]
public class ParseSqlTest
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
                  @level2name = N'addr';            -- 第 3 級名稱（欄位名稱)
                  """;
        
        var rc = ParseSql(sql);
        
        ThenSqlStatement(rc, new SqlSpAddExtendedProperty()
        {
            Name = "MS_Description",
            Value = "hello",
            Level0Type = "SCHEMA",
            Level0Name = "dbo",
            Level1Type = "TABLE",
            Level1Name = "customer",
            Level2Type = "COLUMN",
            Level2Name = "addr",
        });
    }
    
    [Test]
    public void CreateTable()
    {
        var sql = $"""
                   CREATE TABLE Persons (
                   id int,
                   LastName varchar(50),
                   Money decimal(10,3),
                   [name] [int] IDENTITY(1,1) NOT NULL,
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

        ThenSqlStatement(rc, new CreateTableStatement
        {
            TableName = "Persons",
            Columns =
            [
                new ColumnDefinition { ColumnName = "id", DataType = "int" },
                new ColumnDefinition { ColumnName = "LastName", DataType = "varchar", Size = 50 },
                new ColumnDefinition { ColumnName = "Money", DataType = "decimal", Size = 10, Scale = 3 },
                new ColumnDefinition
                {
                    ColumnName = "[name]", DataType = "[int]",
                    Identity = new SqlIdentity()
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
                    Size = 19,
                    Scale = 6,
                    Constraints = [
                        new SqlConstraintDefault
                        {
                            ConstraintName = "[DF_CheckSum]",
                            Value = "(0)",
                        }
                    ]
                }
            ],
            Constraints = [
                new SqlConstraint
                {
                    ConstraintName = "[PK_AcceptedBets]",
                    ConstraintType = "PRIMARY KEY",
                    Clustered = "CLUSTERED",
                    Columns = [
                        new SqlColumnConstraint
                        {
                            ColumnName = "[MatchResultID]",
                            Order = "ASC"
                        }
                    ],
                    WithToggles = [
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
    public void Select()
    {
        var sql = $"""
                   SELECT Id, Name 
                   FROM Persons
                   WHERE Id = 1;
                   """;

        var rc = ParseSql(sql);

        ThenSqlStatement(rc, new SelectStatement
        {
            Columns =
            [
                new SelectColumn() { ColumnName = "Id" },
                new SelectColumn() { ColumnName = "Name" },
            ],
            From = new SelectFrom
            {
                FromTableName = "Persons"
            },
            Where = new SqlWhereExpression
            {
                Left = new SqlFieldExpression
                {
                    FieldName = "Id",
                },
                Operation = "=",
                Right = new SqlIntValueExpression
                {
                    Value = 1
                }
            }
        });
    }

    private static void ThenSqlStatement<T>(Either<ISqlExpression, ParseError> rc, T expectedSqlStatement)
        where T : ISqlExpression
    {
        rc.Switch(statement =>
            {
                var castedStatement = (T)statement;
                castedStatement.Should().BeEquivalentTo(expectedSqlStatement);
            },
            error => throw error
        );
    }

    private static Either<ISqlExpression, ParseError> ParseSql(string sql)
    {
        var p = new SqlParser(sql);
        var rc = p.Parse();
        return rc;
    }
}