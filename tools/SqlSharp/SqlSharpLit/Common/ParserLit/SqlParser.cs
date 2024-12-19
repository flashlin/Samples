using System.Diagnostics;
using System.Runtime.InteropServices.JavaScript;
using System.Text.RegularExpressions;
using T1.SqlSharp;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlParser
{
    private const string ConstraintKeyword = "CONSTRAINT";

    private static readonly string[] ReservedWords =
    [
        "FROM", "SELECT", "JOIN", "LEFT", "UNION", "ON", "GROUP", "WITH",
        "WHERE", "UNPIVOT", "PIVOT", "FOR", "AS"
    ];

    private static string[] DataTypes =
    {
        "BIGINT", "INT", "SMALLINT", "TINYINT", "BIT", "DECIMAL", "NUMERIC", "MONEY", "SMALLMONEY",
        "FLOAT", "REAL", "DATE", "DATETIME", "DATETIME2", "DATETIMEOFFSET", "TIME", "CHAR", "VARCHAR",
        "TEXT", "NCHAR", "NVARCHAR", "NTEXT", "BINARY", "VARBINARY", "IMAGE", "UNIQUEIDENTIFIER", "XML",
        "CURSOR", "TIMESTAMP", "ROWVERSION", "HIERARCHYID", "GEOMETRY", "GEOGRAPHY", "SQL_VARIANT"
    };

    private readonly StringParser _text;

    public SqlParser(string text)
    {
        _text = new StringParser(text);
    }

    public IEnumerable<ISqlExpression> Extract()
    {
        while (!_text.IsEnd())
        {
            var rc = Parse();
            if (rc.HasValue)
            {
                yield return rc.ResultValue;
            }
            else
            {
                _text.ReadUntil(c => c == '\n');
            }
        }
    }

    public string GetRemainingText()
    {
        return _text.GetRemainingText();
    }
    
    public string GetPreviousText(int offset)
    {
        return _text.GetPreviousText(offset);
    }

    public static ParseResult<ISqlExpression> Parse(string sql)
    {
        var p = new SqlParser(sql);
        return p.Parse();
    }

    public ParseResult<ISqlExpression> Parse()
    {
        if (Try(ParseCreateTableStatement, out var createTableStatement))
        {
            return createTableStatement.Result;
        }

        if (Try(ParseSelectStatement, out var selectStatement))
        {
            return selectStatement.Result;
        }

        if (Try(ParseExecSpAddExtendedProperty, out var execSpAddExtendedProperty))
        {
            return execSpAddExtendedProperty.Result;
        }

        if (Try(ParseSetValueStatement, out var setValueStatement))
        {
            return setValueStatement.Result;
        }

        return CreateParseError("Unknown statement");
    }

    public IEnumerable<ISqlExpression> ExtractStatements()
    {
        while (!_text.IsEnd())
        {
            var rc = Parse();
            if (rc.HasError)
            {
                _text.ReadNextSqlToken();
                continue;
            }

            if (rc.Result != null)
            {
                yield return rc.ResultValue;
            }
        }
    }

    public ParseResult<SelectType> Parse_SelectTypeClause()
    {
        var rc = Or(Keywords("ALL"), Keywords("DISTINCT"))();
        if (rc.HasError)
        {
            return rc.Error;
        }

        if (rc is not { HasValue: true, Result: not null })
        {
            return CreateParseResult(SelectType.All);
        }

        var selectType = rc.Result.Value.ToUpper() switch
        {
            "ALL" => SelectType.All,
            "DISTINCT" => SelectType.Distinct,
            _ => SelectType.All
        };
        return CreateParseResult(selectType);
    }

    private ParseResult<SqlChangeTableChanges> Parse_ChangeTableChanges()
    {
        if (!TryKeywords(["CHANGETABLE"], out var startSpan))
        {
            return NoneResult<SqlChangeTableChanges>();
        }

        if (!TryMatch("(", out _))
        {
            return CreateParseError("Expected (");
        }

        if (!TryKeyword("CHANGES", out _))
        {
            return CreateParseError("Expected CHANGES");
        }

        var tableName = _text.ReadSqlIdentifier();
        if (!TryMatch(",", out _))
        {
            return CreateParseError("Expected ,");
        }

        var syncVersion = ParseArithmeticExpr();
        if (syncVersion.HasError)
        {
            return syncVersion.Error;
        }

        if (!TryMatch(")", out _))
        {
            return CreateParseError("Expected )");
        }

        var alias = ParseAliasExpr();
        if (alias.HasError)
        {
            return alias.Error;
        }

        return new SqlChangeTableChanges
        {
            Span = _text.CreateSpan(startSpan),
            TableName = tableName.Word,
            LastSyncVersion = syncVersion.ResultValue,
            Alias = alias.Result?.Name ?? string.Empty
        };
    }

    private ParseResult<SqlGroupByClause> ParseGroupByClause()
    {
        if (!TryKeywords(["GROUP", "BY"], out var startSpan))
        {
            return NoneResult<SqlGroupByClause>();
        }

        var groupByColumns = ParseWithComma(ParseArithmeticExpr);
        if (groupByColumns.HasError)
        {
            return groupByColumns.Error;
        }

        return new SqlGroupByClause
        {
            Span = _text.CreateSpan(startSpan),
            Columns = groupByColumns.ResultValue
        };
    }

    public ParseResult<SqlTopClause> Parse_TopClause()
    {
        if (!TryKeyword("TOP", out var startSpan))
        {
            return NoneResult<SqlTopClause>();
        }

        var expression = Parse_Value_As_DataType();
        if (expression.HasError)
        {
            return expression.Error;
        }

        if (expression.Result == null)
        {
            return CreateParseError("Expected TOP expression");
        }

        var topClause = new SqlTopClause()
        {
            Span = _text.CreateSpan(startSpan),
            Expression = expression.ResultValue
        };
        if (TryKeyword("PERCENT", out _))
        {
            topClause.IsPercent = true;
        }

        if (TryKeywords(["WITH", "TIES"], out _))
        {
            topClause.IsWithTies = true;
        }

        topClause.Span = _text.CreateSpan(startSpan);
        return CreateParseResult(topClause);
    }

    public ParseResult<ISqlExpression> ParseValue()
    {
        if (TryKeyword("NOT", out var notSpan))
        {
            var value = ParseValue();
            if (value.HasError)
            {
                return value.Error;
            }

            return new SqlNotExpression
            {
                Span = _text.CreateSpan(notSpan),
                Value = value.ResultValue
            };
        }

        if (TryKeyword("EXISTS", out var existsSpan))
        {
            var query = ParseParenthesesWith(ParseSelectStatement);
            if (query.HasError)
            {
                return query.Error;
            }

            return new SqlExistsExpression
            {
                Span = _text.CreateSpan(existsSpan),
                Query = query.ResultValue.Inner
            };
        }

        if (Try(Parse_Values, out var values))
        {
            return values.ResultValue;
        }

        if (TryMatch("(", out var openParenthesis))
        {
            var value = ParseArithmeticExpr();
            if (value.HasError)
            {
                return value.Error;
            }

            if (!TryMatch(")", out var closeParenthesis))
            {
                return CreateParseError("Expected )");
            }

            return new SqlGroup
            {
                Span = _text.CreateSpan(openParenthesis),
                Inner = value.ResultValue
            };
        }

        if (Try(ParseUnaryExpr, out var unaryExpr))
        {
            return unaryExpr.ResultValue;
        }

        if (TryMatch("*", out var starSpan))
        {
            return new SqlValue
            {
                Span = starSpan,
                Value = "*"
            };
        }

        if (Try(Parse_DistinctExpr, out var distinctExpr))
        {
            return distinctExpr.ResultValue;
        }

        if (TryKeyword("NULL", out var nullSpan))
        {
            return new SqlNullValue()
            {
                Span = nullSpan
            };
        }

        if (Try(ParseNumberValue, out var numberValue))
        {
            return numberValue.ResultValue;
        }

        if (Try(Parse_NegativeValue, out var negativeValue))
        {
            return negativeValue.ResultValue;
        }

        if (Try(ParseSqlQuotedString, out var quotedString))
        {
            return quotedString.ResultValue;
        }

        if (IsPeekKeywords("SELECT"))
        {
            var subSelect = ParseSelectStatement();
            if (subSelect.HasError)
            {
                return subSelect.Error;
            }

            return subSelect.ResultValue;
        }

        if (Try(ParseCaseClause, out var caseExpr))
        {
            return caseExpr.ResultValue;
        }

        if (Try(ParseRankClause, out var rankClause))
        {
            return rankClause.ResultValue;
        }

        if (Try(ParseFunctionCall, out var function))
        {
            return function.ResultValue;
        }

        if (TryReadSqlIdentifier(out var identifierSpan))
        {
            return new SqlFieldExpr
            {
                Span = identifierSpan,
                FieldName = identifierSpan.Word
            };
        }

        if (Try(ParseTableName, out var tableName))
        {
            return tableName.ResultValue;
        }

        return NoneResult<ISqlExpression>();
    }

    private ParseResult<SqlDistinct> Parse_DistinctExpr()
    {
        if (!TryKeyword("DISTINCT", out var startSpan))
        {
            return NoneResult<SqlDistinct>();
        }

        var value = ParseArithmeticExpr();
        if (value.HasError)
        {
            return value.Error;
        }

        return new SqlDistinct()
        {
            Span = _text.CreateSpan(startSpan),
            Value = value.ResultValue
        };
    }

    public ParseResult<ISqlExpression> Parse_Value_As_DataType()
    {
        var value = ParseValue();
        if (value.HasError)
        {
            return value.Error;
        }

        if (value.Result == null)
        {
            return NoneResult<ISqlExpression>();
        }

        if (Try(ParseOverPartitionByClause, out var overPartitionByClause))
        {
            overPartitionByClause.ResultValue.Field = value.ResultValue;
            value = overPartitionByClause.ResultValue;
        }

        if (Try(ParseOverOrderByClause, out var overOrderByClause))
        {
            overOrderByClause.ResultValue.Field = value.ResultValue;
            value = overOrderByClause.ResultValue;
        }

        if (TryKeyword("AS", out var asSpan))
        {
            var dataType = Or<ISqlExpression>(
                Parse_DataTypeWithSize, 
                ParseSqlQuotedString, Parse_SqlIdentifier)();
            return new SqlAsExpr
            {
                Span = _text.CreateSpan(asSpan),
                Instance = value.ResultValue,
                As = dataType.ResultValue
            };
        }

        return value;
    }

    public ParseResult<ISqlExpression> ParseArithmetic_AdditionOrSubtraction(
        Func<ParseResult<ISqlExpression>> parseTerm)
    {
        var left = parseTerm();
        while (PeekSymbolString(1).Equals("+") || PeekSymbolString(1).Equals("-"))
        {
            var op = ReadSymbolString(1);
            var right = parseTerm();
            left = CreateParseResult(new SqlArithmeticBinaryExpr
            {
                Span = _text.CreateSpan(left.ResultValue.Span, right.ResultValue.Span),
                Left = left.ResultValue,
                Operator = op.ToArithmeticOperator(),
                Right = right.ResultValue
            }).To<ISqlExpression>();
        }

        return left;
    }

    public ParseResult<ISqlExpression> ParseArithmetic_Bitwise(Func<ParseResult<ISqlExpression>> parseTerm)
    {
        var left = parseTerm();
        while (PeekSymbolString(1).Equals("&") || PeekSymbolString(1).Equals("|") || PeekSymbolString(1).Equals("^"))
        {
            var op = ReadSymbolString(1);
            var right = parseTerm();
            left = new SqlArithmeticBinaryExpr
            {
                Span = _text.CreateSpan(left.ResultValue.Span, right.ResultValue.Span),
                Left = left.ResultValue,
                Operator = op.ToArithmeticOperator(),
                Right = right.ResultValue,
            };
        }

        return left;
    }

    public ParseResult<ISqlExpression> ParseArithmetic_MultiplicationOrDivision(
        Func<ParseResult<ISqlExpression>> parseTerm)
    {
        var left = parseTerm();
        while (TryPeekSymbolContains(1, ["*", "/", "%"], out _))
        {
            var op = ReadSymbolString(1);
            var right = parseTerm();
            left = new SqlArithmeticBinaryExpr
            {
                Span = _text.CreateSpan(left.ResultValue.Span, right.ResultValue.Span),
                Left = left.ResultValue,
                Operator = op.ToArithmeticOperator(),
                Right = right.ResultValue,
            };
        }

        return left;
    }

    public ParseResult<ISqlExpression> ParseArithmeticExpr()
    {
        return Parse_SearchCondition(
            () => Parse_ConditionExpr(
                () => ParseArithmetic_AdditionOrSubtraction(
                    () => ParseArithmetic_MultiplicationOrDivision(
                        () => ParseArithmetic_Bitwise(
                            ParseArithmetic_Primary
                        )
                    )
                )
            ));
    }


    public ParseResult<List<ISqlExpression>> ParseCreateTableColumns()
    {
        var columns = new List<ISqlExpression>();
        do
        {
            SkipWhiteSpace();

            // 一開始就定義這些 關鍵字表示不是 column  
            if (IsAny(PeekKeywords("CONSTRAINT"), PeekKeywords("PRIMARY", "KEY"), PeekKeywords("UNIQUE"),
                    PeekKeywords("FOREIGN", "KEY")))
            {
                break;
            }

            var columnDefinition = Or<ISqlExpression>(ParseComputedColumnDefinition, ParseColumnDefinition)();
            if (columnDefinition.HasError)
            {
                return columnDefinition.Error;
            }

            if (columnDefinition.Result != null)
            {
                columns.Add(columnDefinition.Result);
            }

            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();

            // 怪異的 SQL 語法: 允許逗號後面沒有東西 遇到 ) 直接結束 
            if (_text.PeekChar() == ')')
            {
                break;
            }
        } while (!_text.IsEnd());

        return CreateParseResult(columns);
    }

    public ParseResult<SqlCreateTableExpression> ParseCreateTableStatement()
    {
        if (!TryKeywords(["CREATE", "TABLE"], out var startSpan))
        {
            return NoneResult<SqlCreateTableExpression>();
        }

        var tableName = _text.ReadSqlIdentifier();
        if (!TryMatch("(", out var openParenthesis))
        {
            return CreateParseError("Expected (");
        }

        var createTableStatement = new SqlCreateTableExpression()
        {
            Span = _text.CreateSpan(startSpan),
            TableName = tableName.Word,
        };

        while (!_text.IsEnd())
        {
            var tableColumnsResult = ParseCreateTableColumns();
            if (tableColumnsResult.HasError)
            {
                return tableColumnsResult.Error;
            }

            var tableColumns = tableColumnsResult.ResultValue;
            if (tableColumns.Count > 0)
            {
                createTableStatement.Columns.AddRange(tableColumns);
                continue;
            }

            var tableConstraintsResult = ParseWithComma(ParseTableConstraint);
            if (tableConstraintsResult.HasError)
            {
                return tableConstraintsResult.Error;
            }

            var tableConstraints = tableConstraintsResult.ResultValue;
            if (tableConstraints.Count > 0)
            {
                createTableStatement.Constraints.AddRange(tableConstraints);
                continue;
            }

            break;
        }

        if (!TryMatch(")", out var closeParenthesis))
        {
            return CreateParseError("ParseCreateTableStatement Expected )");
        }

        SkipStatementEnd();

        createTableStatement.Span = _text.CreateSpan(startSpan);
        return CreateParseResult(createTableStatement);
    }

    public ParseResult<SqlSpAddExtendedPropertyExpression> ParseExecSpAddExtendedProperty()
    {
        if (!TryKeywords(["EXEC", "SP_AddExtendedProperty"], out var startSpan))
        {
            return NoneResult<SqlSpAddExtendedPropertyExpression>();
        }

        var parameters = ParseWithComma(ParseParameterValueOrAssignValue);
        if (parameters.HasError)
        {
            return parameters.Error;
        }

        if (parameters.ResultValue.Count != 8)
        {
            return CreateParseError("Expected 8 parameters");
        }

        var p = parameters.ResultValue;

        var sqlSpAddExtendedProperty = new SqlSpAddExtendedPropertyExpression
        {
            Span = _text.CreateSpan(startSpan),
            Name = p[0].Value,
            Value = p[1].Value,
            Level0Type = p[2].Value,
            Level0Name = p[3].Value,
            Level1Type = p[4].Value,
            Level1Name = p[5].Value,
            Level2Type = p[6].Value,
            Level2Name = p[7].Value
        };
        return CreateParseResult(sqlSpAddExtendedProperty);
    }

    public ParseResult<SqlConstraintForeignKey> ParseForeignKeyExpression()
    {
        if (!TryKeywords(["FOREIGN", "KEY"], out var startSpan))
        {
            return NoneResult<SqlConstraintForeignKey>();
        }

        var columnsResult = ParseColumnsAscDesc();
        if (columnsResult.HasError)
        {
            return columnsResult.Error;
        }

        var columns = columnsResult.ResultValue;
        if (!TryKeyword("REFERENCES", out _))
        {
            return CreateParseError("Expected REFERENCES");
        }

        var tableName = _text.ReadSqlIdentifier();
        if (tableName.Length == 0)
        {
            return CreateParseError("Expected reference table name");
        }

        var refColumn = string.Empty;
        if (TryMatch("(", out var openParenthesis))
        {
            refColumn = _text.ReadSqlIdentifier().Word;
            if (!TryMatch(")", out var closeParenthesis))
            {
                return CreateParseError("Expected )");
            }
        }

        var onDelete = ReferentialAction.NoAction;
        if (TryKeywords(["ON", "DELETE"], out _))
        {
            var rc = ParseReferentialAction();
            if (rc.HasError)
            {
                return rc.Error;
            }

            onDelete = rc.Result;
        }

        var onUpdate = ReferentialAction.NoAction;

        if (TryKeywords(["ON", "UPDATE"], out _))
        {
            var rc = ParseReferentialAction();
            if (rc.HasError)
            {
                return rc.Error;
            }

            onUpdate = rc.Result;
        }

        var notForReplication = TryKeywords(["NOT", "FOR", "REPLICATION"], out _);
        return CreateParseResult(new SqlConstraintForeignKey
        {
            Span = _text.CreateSpan(startSpan),
            Columns = columns,
            ReferencedTableName = tableName.Word,
            RefColumn = refColumn,
            OnDeleteAction = onDelete,
            OnUpdateAction = onUpdate,
            NotForReplication = notForReplication,
        });
    }

    public ParseResult<SelectStatement> ParseSelectStatement()
    {
        if (!TryKeyword("SELECT", out var startSpan))
        {
            return NoneResult<SelectStatement>();
        }

        var selectStatement = new SelectStatement();

        var selectTypeClause = Parse_SelectTypeClause();
        if (selectTypeClause.HasError)
        {
            return selectTypeClause.Error;
        }

        selectStatement.SelectType = selectTypeClause.ResultValue;

        var topClause = Parse_TopClause();
        if (topClause.HasError)
        {
            return topClause.Error;
        }

        if (topClause.Result != null)
        {
            selectStatement.Top = topClause.Result;
        }

        var columns = Parse_SelectColumns();
        if (columns.HasError)
        {
            return columns.Error;
        }

        selectStatement.Columns = columns.ResultValue;

        if (TryKeyword("FROM", out _))
        {
            var tableSources = Or(Parse_FromGroupFromTableSources, Parse_FromTableSources)();
            if (tableSources.HasError)
            {
                return tableSources.Error;
            }

            selectStatement.FromSources = tableSources.ResultValue;
        }

        if (Try(ParsePivotClause, out var pivotClause))
        {
            selectStatement.FromSources.Add(pivotClause.ResultValue);
        }

        if (Try(ParseUnpivotClause, out var unpivotClause))
        {
            selectStatement.FromSources.Add(unpivotClause.ResultValue);
        }

        if (TryKeyword("WHERE", out _))
        {
            var rc = Parse_WhereExpression();
            if (rc.HasError)
            {
                return rc.Error;
            }

            selectStatement.Where = rc.Result;
        }

        if (Try(ParseGroupByClause, out var groupByClause))
        {
            selectStatement.GroupBy = groupByClause.Result;
        }

        var orderByClause = ParseOrderByClause();
        if (orderByClause.HasError)
        {
            return orderByClause.Error;
        }

        selectStatement.OrderBy = orderByClause.Result;

        if (Try(ParseForXmlClause, out var forXmlClause))
        {
            selectStatement.ForXml = forXmlClause.ResultValue;
        }

        if (Try(ParseUnionSelectClauseList, out var unionSelectClauseList))
        {
            selectStatement.Unions = unionSelectClauseList.ResultValue;
        }

        if (Try(ParseHavingClause, out var havingClause))
        {
            selectStatement.Having = havingClause.Result;
        }

        SkipStatementEnd();
        selectStatement.Span = _text.CreateSpan(startSpan);
        return CreateParseResult(selectStatement);
    }

    private ParseResult<List<ISqlExpression>> Parse_FromGroupFromTableSources()
    {
        var startPosition = _text.Position;
        if (!TryMatch("(", out _))
        {
            return NoneResult<List<ISqlExpression>>();
        }

        if (IsPeekKeywords("SELECT"))
        {
            _text.Position = startPosition;
            return NoneResult<List<ISqlExpression>>();
        }

        var tableSources = Parse_FromTableSources();
        if (!TryMatch(")", out _))
        {
            return CreateParseError("Expected )");
        }

        return tableSources.ResultValue;
    }

    private ParseResult<ISqlForXmlClause> ParseForXmlClause()
    {
        if (Try(ParseForXmlPathClause, out var forXmlPathClause))
        {
            return forXmlPathClause.ResultValue;
        }

        if (Try(ParseForXmlAutoClause, out var forXmlAutoClause))
        {
            return forXmlAutoClause.ResultValue;
        }

        return NoneResult<ISqlForXmlClause>();
    }

    private ParseResult<SqlPivotClause> ParsePivotClause()
    {
        if (!TryKeyword("PIVOT", out var startSpan))
        {
            return NoneResult<SqlPivotClause>();
        }

        if (!TryMatch("(", out var openParenthesis))
        {
            return CreateParseError("Expected (");
        }

        var newColumn = ParseValue();
        if (newColumn.HasError)
        {
            return newColumn.Error;
        }

        if (!TryKeyword("FOR", out _))
        {
            return CreateParseError("Expected FOR");
        }

        var forSource = ParseValue();
        if (forSource.HasError)
        {
            return forSource.Error;
        }

        if (!TryKeyword("IN", out _))
        {
            return CreateParseError("Expected IN");
        }

        var inColumns = ParseParenthesesWithComma(ParseValue);

        if (!TryMatch(")", out var closeParenthesis))
        {
            return CreateParseError("Expected )");
        }

        var alias = ParseAliasExpr();

        return new SqlPivotClause
        {
            Span = _text.CreateSpan(startSpan),
            NewColumn = newColumn.ResultValue,
            ForSource = forSource.ResultValue,
            InColumns = inColumns.ResultValue,
            AliasName = alias.ResultValue.Name
        };
    }

    private ParseResult<SqlUnpivotClause> ParseUnpivotClause()
    {
        if (!TryKeyword("UNPIVOT", out var startSpan))
        {
            return NoneResult<SqlUnpivotClause>();
        }

        if (!TryMatch("(", out var openParenthesis))
        {
            return CreateParseError("Expected (");
        }

        var newColumn = ParseValue();
        if (newColumn.HasError)
        {
            return newColumn.Error;
        }

        if (!TryKeyword("FOR", out _))
        {
            return CreateParseError("Expected FOR");
        }

        var forSource = ParseValue();
        if (forSource.HasError)
        {
            return forSource.Error;
        }

        if (!TryKeyword("IN", out _))
        {
            return CreateParseError("Expected IN");
        }

        var inColumns = ParseParenthesesWithComma(ParseValue);

        if (!TryMatch(")", out var closeParenthesis))
        {
            return CreateParseError("Expected )");
        }

        var alias = ParseAliasExpr();

        return CreateParseResult(new SqlUnpivotClause
        {
            Span = _text.CreateSpan(startSpan),
            NewColumn = newColumn.ResultValue,
            ForSource = forSource.ResultValue,
            InColumns = inColumns.ResultValue,
            AliasName = alias.ResultValue.Name
        });
    }

    private ParseResult<SqlForXmlPathClause> ParseForXmlPathClause()
    {
        if (!TryKeywords(["FOR", "XML", "PATH"], out var startSpan))
        {
            return NoneResult<SqlForXmlPathClause>();
        }

        var forXmlClause = new SqlForXmlPathClause();
        if (TryMatch("(", out var openParenthesis))
        {
            forXmlClause.PathName = Parse_QuotedString().ResultValue.Value;
            MatchSymbol(")");
        }

        forXmlClause.CommonDirectives = Parse_ForXmlRootDirectives();
        forXmlClause.Span = _text.CreateSpan(startSpan);
        return forXmlClause;
    }

    private ParseResult<SqlForXmlAutoClause> ParseForXmlAutoClause()
    {
        if (!TryKeywords(["FOR", "XML", "AUTO"], out var startSpan))
        {
            return NoneResult<SqlForXmlAutoClause>();
        }

        var forXmlClause = new SqlForXmlAutoClause
        {
            CommonDirectives = Parse_ForXmlRootDirectives(),
            Span = _text.CreateSpan(startSpan),
        };
        return forXmlClause;
    }

    private List<SqlForXmlRootDirective> Parse_ForXmlRootDirectives()
    {
        var directives = new List<SqlForXmlRootDirective>();
        if (!TryMatch(",", out var commaSpan))
        {
            return directives;
        }

        var elements = ParseWithComma(() =>
        {
            if (TryKeyword("ROOT", out var span))
            {
                var rootName = ParseWithParentheses(ParseValue);
                return new SqlForXmlRootDirective
                {
                    Span = _text.CreateSpan(span),
                    RootName = rootName.ResultValue
                };
            }

            return NoneResult<SqlForXmlRootDirective>();
        });
        directives.AddRange(elements.ResultValue);
        return directives;
    }

    private ParseResult<List<SqlUnionSelect>> ParseUnionSelectClauseList()
    {
        var unionSelectList = new List<SqlUnionSelect>();
        do
        {
            var unionSelect = Parse_UnionSelect();
            if (unionSelect.HasError)
            {
                return unionSelect.Error;
            }

            if (unionSelect.Result == null)
            {
                break;
            }

            unionSelectList.Add(unionSelect.ResultValue);
        } while (true);

        return CreateParseResult(unionSelectList);
    }

    public void SkipStatementEnd()
    {
        var ch = _text.PeekChar();
        if (ch == ';')
        {
            _text.ReadChar();
        }
    }

    private ParseResult<SqlUnionSelect> Parse_UnionSelect()
    {
        var isAll = false;
        var startSpan = new TextSpan();
        if (!TryKeywords(["UNION", "ALL"], out startSpan))
        {
            if (!TryKeyword("UNION", out startSpan))
            {
                return NoneResult<SqlUnionSelect>();
            }
        }
        else
        {
            isAll = true;
        }

        var select = ParseGroupOr(ParseSelectStatement);
        if (select.HasError)
        {
            return select.Error;
        }

        return new SqlUnionSelect
        {
            Span = _text.CreateSpan(startSpan),
            IsAll = isAll,
            SelectStatement = select.ResultValue,
        };
    }

    private ParseResult<ISqlExpression> ParseGroupOr<T>(Func<ParseResult<T>> parseFn)
        where T : ISqlExpression
    {
        if (TryMatch("(", out var openSpan))
        {
            var inner = parseFn();
            if (inner.HasError)
            {
                return inner.Error;
            }

            var closeSpan = MatchSymbol(")");
            return new SqlGroup()
            {
                Span = _text.CreateSpan(openSpan, closeSpan),
                Inner = inner.ResultValue
            };
        }

        return parseFn().To<ISqlExpression>();
    }

    public bool Try<T>(Func<ParseResult<T>> parseFunc, out ParseResult<T> result)
    {
        var localResult = parseFunc();
        if (localResult.HasError)
        {
            result = localResult;
            return true;
        }

        if (localResult.Result == null)
        {
            result = localResult;
            return false;
        }

        result = localResult;
        return true;
    }

    private ParseError CreateParseError(string error)
    {
        return new ParseError(error)
        {
            Offset = _text.Position
        };
    }

    private ParseResult<T> CreateParseResult<T>(T result)
    {
        return new ParseResult<T>(result);
    }

    private T? GetResult<T>(Func<ParseResult<T>> parseFn)
    {
        var result = parseFn();
        return result.Result;
    }

    private bool IsAny<T>(params Func<ParseResult<T>>[] parseFnList)
    {
        var span = Or(parseFnList)();
        if (span.HasError)
        {
            return false;
        }

        return span.Result != null;
    }

    private bool IsPeek<T>(Func<ParseResult<T>> parseFn)
    {
        SkipWhiteSpace();
        var tmpPosition = _text.Position;
        var rc = parseFn();
        var isSuccess = rc.Result != null;
        _text.Position = tmpPosition;
        return isSuccess;
    }

    private ParseResult<SqlUnaryExpr> ParseUnaryExpr()
    {
        if (TryMatch("~", out var startSpan))
        {
            var expr = ParseArithmeticExpr();
            if (expr.HasError)
            {
                return expr.Error;
            }

            return new SqlUnaryExpr
            {
                Span = _text.CreateSpan(startSpan, expr.ResultValue.Span),
                Operator = UnaryOperator.BitwiseNot,
                Operand = expr.ResultValue
            };
        }

        return NoneResult<SqlUnaryExpr>();
    }

    private ParseResult<SqlRankClause> ParseRankClause()
    {
        var startPosition = _text.Position;
        if (!TryKeyword("RANK", out var startSpan))
        {
            return NoneResult<SqlRankClause>();
        }

        if (!TryMatch("()", out _))
        {
            _text.Position = startPosition;
            return NoneResult<SqlRankClause>();
        }

        if (!TryKeyword("OVER", out _))
        {
            return CreateParseError("Expected OVER");
        }

        if (!TryMatch("(", out _))
        {
            return CreateParseError("Expected (");
        }

        var partitionBy = ParsePartitionBy().Result;
        var orderBy = ParseOrderByClause().ResultValue;
        if (!TryMatch(")", out _))
        {
            return CreateParseError("Expected )");
        }

        return new SqlRankClause
        {
            Span = _text.CreateSpan(startSpan),
            PartitionBy = partitionBy,
            OrderBy = orderBy
        };
    }

    private ParseResult<SqlOverOrderByClause> ParseOverOrderByClause()
    {
        if (!TryKeywords(["OVER"], out var startSpan))
        {
            return NoneResult<SqlOverOrderByClause>();
        }

        if (!TryMatch("(", out _))
        {
            return NoneResult<SqlOverOrderByClause>();
        }

        var orderByClause = ParseOrderByClause();
        if (orderByClause.HasError)
        {
            return orderByClause.Error;
        }

        var orderColumns = orderByClause.Result?.Columns ?? [];
        if (!TryMatch(")", out _))
        {
            return CreateParseError("Expected )");
        }

        return new SqlOverOrderByClause
        {
            Span = _text.CreateSpan(startSpan),
            Columns = orderColumns
        };
    }

    private ParseResult<SqlOverPartitionByClause> ParseOverPartitionByClause()
    {
        if (!TryKeywords(["OVER"], out var startSpan))
        {
            return NoneResult<SqlOverPartitionByClause>();
        }

        if (!TryMatch("(", out _))
        {
            _text.Position = startSpan.Offset;
            return NoneResult<SqlOverPartitionByClause>();
        }

        if (!TryKeywords(["PARTITION", "BY"], out _))
        {
            _text.Position = startSpan.Offset;
            return NoneResult<SqlOverPartitionByClause>();
        }

        var partitionBy = Parse_SqlIdentifier().ResultValue;
        var orderBy = ParseOrderByClause();
        if (orderBy.HasError)
        {
            return orderBy.Error;
        }

        if (orderBy.Result == null)
        {
            _text.Position = startSpan.Offset;
            return NoneResult<SqlOverPartitionByClause>();
        }

        var orderColumns = orderBy.ResultValue.Columns;
        if (!TryMatch(")", out _))
        {
            return CreateParseError("Expected )");
        }

        return new SqlOverPartitionByClause()
        {
            Span = _text.CreateSpan(startSpan),
            By = partitionBy,
            Columns = orderColumns
        };
    }

    private ParseResult<SqlPartitionBy> ParsePartitionBy()
    {
        if (!TryKeywords(["PARTITION", "BY"], out var startSpan))
        {
            return NoneResult<SqlPartitionBy>();
        }

        var columns = ParseWithComma(ParseValue);
        if (columns.HasError)
        {
            return columns.Error;
        }

        return CreateParseResult(new SqlPartitionBy
        {
            Span = _text.CreateSpan(startSpan),
            Columns = columns.ResultValue
        });
    }

    private bool IsPeekKeywords(params string[] keywords)
    {
        var keywordsResult = PeekKeywords(keywords)();
        if (keywordsResult.Result != null && keywordsResult.Result.Value.Length != 0)
        {
            return true;
        }

        return false;
    }

    private bool IsPeekMatch(string expected)
    {
        SkipWhiteSpace();
        var tmpPosition = _text.Position;
        var isSuccess = _text.TryMatch(expected, out _);
        _text.Position = tmpPosition;
        return isSuccess;
    }

    private Func<ParseResult<SqlToken>> Keywords(params string[] keywords)
    {
        return () => ParseKeywords(keywords);
    }


    private TextSpan MatchSymbol(string expected)
    {
        SkipWhiteSpace();
        return _text.MatchSymbol(expected);
    }

    private ParseResult<T> NoneResult<T>()
    {
        return new ParseResult<T>(default(T));
    }

    private SqlFunctionExpression NormalizeFunctionName(SqlFunctionExpression function)
    {
        if (function.FunctionName.ToUpper() == "CONVERT")
        {
            var p0 = function.Parameters[0];
            if (p0.SqlType == SqlType.Field)
            {
                var field = (SqlFieldExpr)p0;
                function.Parameters[0] = new SqlDataTypeWithSize
                {
                    DataTypeName = field.FieldName
                };
            }
        }

        return function;
    }

    private Func<ParseResult<T>> One<T>(params Func<ParseResult<T>>[] parseFnList)
    {
        return () =>
        {
            var rc = Or(parseFnList)();
            if (rc.HasError)
            {
                return rc;
            }

            if (rc.Result != null)
            {
                return rc;
            }

            return CreateParseError("Expected one of the options");
        };
    }

    private Func<ParseResult<T>> Or<T>(params Func<IParseResult>[] parseFnList)
    {
        return () =>
        {
            foreach (var parseFn in parseFnList)
            {
                var rc = parseFn();
                if (rc.HasError)
                {
                    return rc.Error;
                }

                if (rc is { HasResult: true, Object: not null })
                {
                    return CreateParseResult((T)rc.ObjectValue);
                }
            }

            return NoneResult<T>();
        };
    }


    private Func<ParseResult<T>> Or<T>(params Func<ParseResult<T>>[] parseFnList)
    {
        return () =>
        {
            foreach (var parseFn in parseFnList)
            {
                var rc = parseFn();
                if (rc.HasError)
                {
                    return rc.Error;
                }

                if (rc is { HasResult: true, Object: not null })
                {
                    return rc.ResultValue;
                }
            }

            return NoneResult<T>();
        };
    }

    private ParseResult<SqlBetweenValue> Parse_BetweenValue()
    {
        var start = ParseArithmeticExpr();
        if (start.HasError)
        {
            return start.Error;
        }

        if (start.ResultValue.SqlType == SqlType.SearchCondition)
        {
            var searchCondition = (SqlSearchCondition)start.ResultValue;
            var betweenExpr = Sub_Between_And_Expr(searchCondition);
            return new SqlBetweenValue
            {
                Span = _text.CreateSpan(searchCondition.Left.Span, betweenExpr!.Span),
                Start = betweenExpr.Left,
                End = betweenExpr.Right!
            };
        }

        if (!TryKeyword("AND", out _))
        {
            return CreateParseError("Expected AND");
        }

        var end = ParseArithmeticExpr();
        if (end.HasError)
        {
            return end.Error;
        }

        return new SqlBetweenValue
        {
            Span = _text.CreateSpan(start.ResultValue.Span, end.ResultValue.Span),
            Start = start.ResultValue,
            End = end.ResultValue
        };
    }

    private SqlSearchCondition? Sub_Between_And_Expr(ISqlExpression endExpr)
    {
        if (endExpr.SqlType == SqlType.SearchCondition)
        {
            var subSearchExpr = (SqlSearchCondition)endExpr;

            if (subSearchExpr.Left.SqlType != SqlType.SearchCondition)
            {
                return subSearchExpr;
            }

            var betweenExpr = subSearchExpr.Left;
            _text.Position = subSearchExpr.OperatorSpan.Offset;
            return (SqlSearchCondition)betweenExpr;
        }

        return null;
    }

    private ParseResult<SqlWhenThenClause> Parse_Case_WhenClause()
    {
        if (!TryKeyword("WHEN", out var startSpan))
        {
            return NoneResult<SqlWhenThenClause>();
        }

        var whenExpr = ParseArithmeticExpr();
        if (whenExpr.HasError)
        {
            return whenExpr.Error;
        }

        if (!TryKeyword("THEN", out _))
        {
            return CreateParseError("Expected THEN");
        }

        var thenExpr = ParseArithmeticExpr();
        if (thenExpr.HasError)
        {
            return thenExpr.Error;
        }

        return CreateParseResult(new SqlWhenThenClause
        {
            Span = _text.CreateSpan(startSpan),
            When = whenExpr.ResultValue,
            Then = thenExpr.ResultValue
        });
    }

    private ParseResult<SqlCaseClause> ParseCaseClause()
    {
        if (!TryKeyword("CASE", out var startSpan))
        {
            return NoneResult<SqlCaseClause>();
        }

        ISqlExpression? whenExpr = null;
        if (!IsPeekKeywords("WHEN"))
        {
            var whenExprRc = ParseArithmeticExpr();
            if (whenExprRc.HasError)
            {
                return whenExprRc.Error;
            }

            whenExpr = whenExprRc.ResultValue;
        }

        var whenClause = new List<SqlWhenThenClause>();
        do
        {
            if (!IsPeekKeywords("WHEN"))
            {
                break;
            }

            var whenThenExpr = Parse_Case_WhenClause();
            if (whenThenExpr.HasError)
            {
                return whenThenExpr.Error;
            }

            if (whenThenExpr.Result == null)
            {
                break;
            }

            whenClause.Add(whenThenExpr.ResultValue);
        } while (true);

        if (whenClause.Count == 0)
        {
            return CreateParseError("Expected WHEN");
        }

        ISqlExpression? elseClause = null;
        if (TryKeyword("ELSE", out _))
        {
            var elseClauseResult = ParseArithmeticExpr();
            if (elseClauseResult.HasError)
            {
                return elseClauseResult.Error;
            }

            elseClause = elseClauseResult.ResultValue;
        }

        if (!TryKeyword("END", out _))
        {
            return CreateParseError("Expected END");
        }

        return CreateParseResult(new SqlCaseClause
        {
            Span = _text.CreateSpan(startSpan),
            Case = whenExpr,
            WhenThens = whenClause,
            Else = elseClause
        });
    }

    private ParseResult<SelectColumn> Parse_Column_Arithmetic()
    {
        if (Try(ParseArithmeticExpr, out var arithmetic))
        {
            return CreateParseResult(new SelectColumn
            {
                Span = arithmetic.ResultValue.Span,
                Field = arithmetic.ResultValue
            });
        }

        return NoneResult<SelectColumn>();
    }

    private ParseResult<SqlComparisonOperator> Parse_ComparisonOperator()
    {
        var rc = Or(
            Keywords("IS", "NOT"),
            Keywords("NOT", "LIKE"),
            Keywords("IS"),
            Keywords("LIKE"),
            Keywords("IN"),
            Keywords("BETWEEN"),
            Symbol("<>"),
            Symbol("!="),
            Symbol(">="),
            SymbolWithNoncontinuous(">="),
            Symbol("<="),
            SymbolWithNoncontinuous("<="),
            Symbol("="),
            Symbol(">"),
            Symbol("<")
        )();
        if (rc.HasError)
        {
            return rc.Error;
        }

        if (rc.Result == null)
        {
            return NoneResult<SqlComparisonOperator>();
        }

        return new SqlComparisonOperator
        {
            Span = rc.Result.Span,
            Value = rc.Result.Value.ToComparisonOperator()
        };
    }

    private ParseResult<ISqlExpression> Parse_ConditionExpr(Func<ParseResult<ISqlExpression>> parseTerm)
    {
        var left = parseTerm();
        while (Try(Parse_ComparisonOperator, out var comparisonOperator))
        {
            var op = comparisonOperator.ResultValue.Value;
            ISqlExpression? right;
            switch (op)
            {
                case ComparisonOperator.Between:
                    var betweenValue = Parse_BetweenValue();
                    if (betweenValue.HasError)
                    {
                        return betweenValue.Error;
                    }

                    right = betweenValue.ResultValue;
                    break;
                default:
                    right = parseTerm().ResultValue;
                    break;
            }

            left = CreateParseResult(new SqlConditionExpression
            {
                Span = _text.CreateSpan(left.ResultValue.Span, right.Span),
                Left = left.ResultValue,
                ComparisonOperator = op,
                OperatorSpan = comparisonOperator.ResultValue.Span,
                Right = right
            }).To<ISqlExpression>();
        }

        return left;
    }

    private ParseResult<SqlDataSize> Parse_DataSize()
    {
        if (!TryMatch("(", out var openParenthesis))
        {
            return NoneResult<SqlDataSize>();
        }

        var dataSize = new SqlDataSize();
        if (_text.TryKeywordIgnoreCase("MAX", out _))
        {
            dataSize.Size = "MAX";
            _text.MatchSymbol(")");
            dataSize.Span = _text.CreateSpan(openParenthesis);
            return dataSize;
        }

        dataSize.Size = _text.ReadInt().Word;
        if (_text.PeekChar() == ',')
        {
            _text.ReadChar();
            dataSize.Scale = int.Parse(_text.ReadInt().Word);
        }

        if (!TryMatch(")", out var closeParenthesis))
        {
            return CreateParseError("Expected )");
        }

        dataSize.Span = _text.CreateSpan(openParenthesis);
        return dataSize;
    }

    private ParseResult<SqlToken> Parse_DataType()
    {
        var startPosition = _text.Position;
        if (!TryReadSqlIdentifier(out var identifierSpan))
        {
            return NoneResult<SqlToken>();
        }

        if (!DataTypes.Contains(identifierSpan.Word.ToUpper()))
        {
            _text.Position = startPosition;
            return NoneResult<SqlToken>();
        }

        return new SqlToken
        {
            Value = identifierSpan.Word,
            Span = identifierSpan
        };
    }

    private ParseResult<SqlDataTypeWithSize> Parse_DataTypeWithSize()
    {
        var startPosition = _text.Position;
        if (!TryReadSqlIdentifier(out var identifierSpan))
        {
            return NoneResult<SqlDataTypeWithSize>();
        }
        
        if (!DataTypes.Contains(identifierSpan.Word.ToUpper()))
        {
            _text.Position = startPosition;
            return NoneResult<SqlDataTypeWithSize>();
        }

        var dataType = Parse_DataSize();
        if (dataType.HasError)
        {
            return dataType.Error;
        }

        return new SqlDataTypeWithSize()
        {
            DataTypeName = identifierSpan.Word,
            Size = dataType.Result != null ? dataType.ResultValue : new SqlDataSize()
        };
    }

    private ParseResult<ITableSource> Parse_FromTableSource()
    {
        if (TryMatch("(", out var openParenthesis))
        {
            var sub = ParseSelectStatement();
            if (sub.HasError)
            {
                return sub.Error;
            }

            MatchSymbol(")");
            return new SqlInnerTableSource()
            {
                Inner = sub.ResultValue
            };
        }
        
        if(Try(Parse_ChangeTableChanges, out var changeTableChanges))
        {
            return new SqlInnerTableSource()
            {
                Inner = changeTableChanges.ResultValue,
                Alias = changeTableChanges.ResultValue.Alias
            };
        }

        if (Try(ParseFunctionCall, out var function))
        {
            return new SqlFuncTableSource()
            {
                Span = function.ResultValue.Span,
                Function = function.ResultValue
            };
        }

        if (Try(Parse_SqlIdentifier, out var tableName))
        {
            return new SqlTableSource()
            {
                Span = tableName.ResultValue.Span,
                TableName = tableName.ResultValue.FieldName
            };
        }

        return NoneResult<ITableSource>();
    }

    private ParseResult<List<ISqlExpression>> Parse_FromTableSources()
    {
        var allTableSources = new List<ISqlExpression>();
        ParseWithComma(() =>
        {
            var fromTableSources = Parse_FromTableSourcesWithComma();
            var tableSourcesExpr = fromTableSources.ResultValue;
            var joinTableSources = Parse_JoinTableSources();
            if (joinTableSources.HasError)
            {
                return joinTableSources.Error;
            }

            if (joinTableSources.Result != null)
            {
                tableSourcesExpr.AddRange(joinTableSources.ResultValue);
            }

            foreach (var tableSource in tableSourcesExpr)
            {
                allTableSources.Add(tableSource);
            }

            return CreateParseResult(tableSourcesExpr[0]);
        });
        return allTableSources;
    }

    private ParseResult<List<ISqlExpression>> Parse_FromTableSourcesWithComma()
    {
        var fromTableSources = ParseWithComma(() =>
        {
            var tableSource = Or<ISqlExpression>(Parse_TableSourceWithHints, Parse_JoinTableSource)();
            if (tableSource.HasError)
            {
                return tableSource.Error;
            }

            return tableSource;
        });
        return fromTableSources;
    }

    private ParseResult<List<SqlJoinTableCondition>> Parse_JoinTableSources()
    {
        var joinTableSources = new List<SqlJoinTableCondition>();
        do
        {
            var joinTable = Parse_JoinTableSource();
            if (joinTable.HasError)
            {
                return joinTable.Error;
            }

            if (joinTable.Result == null)
            {
                break;
            }

            joinTableSources.Add(joinTable.ResultValue);
        } while (true);

        return joinTableSources;
    }

    private ParseResult<SqlFunctionExpression> ParseFunctionCall()
    {
        var startPosition = _text.Position;
        if (TryReadSqlFunctionName(out var identifier))
        {
            if (TryMatch("(", out var openParenthesis))
            {
                var parameters = ParseWithComma(ParseArithmeticExpr);
                MatchSymbol(")");
                var function = new SqlFunctionExpression
                {
                    Span = _text.CreateSpan(startPosition),
                    FunctionName = identifier.Word,
                    Parameters = parameters.ResultValue
                };
                function = NormalizeFunctionName(function);
                return function;
            }
        }

        _text.Position = startPosition;
        return NoneResult<SqlFunctionExpression>();
    }

    private ParseResult<SqlHavingClause> ParseHavingClause()
    {
        if (!TryKeywords(["HAVING"], out var startSpan))
        {
            return NoneResult<SqlHavingClause>();
        }

        var searchCondition = ParseArithmeticExpr();
        if (searchCondition.HasError)
        {
            return searchCondition.Error;
        }

        return new SqlHavingClause
        {
            Span = _text.CreateSpan(startSpan),
            Condition = searchCondition.ResultValue
        };
    }

    private ParseResult<SqlJoinTableCondition> Parse_JoinTableSource()
    {
        if (TryKeywords(["INNER", "JOIN"], out var innerJoinStartSpan))
        {
            var tableSource = Parse_JoinTableSourceOn();
            tableSource.ResultValue.Span = _text.CreateSpan(innerJoinStartSpan);
            return tableSource.ResultValue;
        }

        if (TryKeyword("JOIN", out var joinStartSpan))
        {
            var tableSource = Parse_JoinTableSourceOn();
            tableSource.ResultValue.Span = _text.CreateSpan(joinStartSpan);
            return tableSource.ResultValue;
        }

        if (TryKeywords(["LEFT", "JOIN"], out var leftJoinStartSpan))
        {
            var tableSource = Parse_JoinTableSourceOn().ResultValue;
            tableSource.JoinType = JoinType.Left;
            tableSource.Span = _text.CreateSpan(leftJoinStartSpan);
            return tableSource;
        }

        if (TryKeywords(["LEFT", "OUTER", "JOIN"], out var leftOuterJoinStartSpan))
        {
            var tableSource = Parse_JoinTableSourceOn().ResultValue;
            tableSource.JoinType = JoinType.Left;
            tableSource.Span = _text.CreateSpan(leftOuterJoinStartSpan);
            return tableSource;
        }

        if (TryKeywords(["RIGHT", "JOIN"], out var rightJoinStartSpan))
        {
            var tableSource = Parse_JoinTableSourceOn().ResultValue;
            tableSource.JoinType = JoinType.Right;
            tableSource.Span = _text.CreateSpan(rightJoinStartSpan);
            return tableSource;
        }

        if (TryKeywords(["RIGHT", "OUTER", "JOIN"], out var rightOuterJoinStartSpan))
        {
            var tableSource = Parse_JoinTableSourceOn().ResultValue;
            tableSource.JoinType = JoinType.Right;
            tableSource.Span = _text.CreateSpan(rightOuterJoinStartSpan);
            return tableSource;
        }

        return NoneResult<SqlJoinTableCondition>();
    }

    private ParseResult<SqlJoinTableCondition> Parse_JoinTableSourceOn()
    {
        var startPosition = _text.Position;
        if (!Try(Parse_TableSourceWithHints, out var tableSource))
        {
            return NoneResult<SqlJoinTableCondition>();
        }

        if (!TryKeyword("ON", out _))
        {
            return CreateParseError("Expected ON");
        }

        var onCondition = ParseArithmeticExpr();
        return new SqlJoinTableCondition()
        {
            Span = _text.CreateSpan(startPosition),
            JoinedTable = tableSource.ResultValue,
            OnCondition = onCondition.ResultValue,
        };
    }

    private ParseResult<SqlLogicalOperator> Parse_LogicalOperator()
    {
        var rc = Or(Keywords("AND"), Keywords("OR"), Keywords("NOT"))();
        if (rc.HasError)
        {
            return rc.Error;
        }

        if (rc.Result == null)
        {
            return NoneResult<SqlLogicalOperator>();
        }

        return new SqlLogicalOperator
        {
            Span = rc.ResultValue.Span,
            Value = rc.Result.Value.ToLogicalOperator(),
        };
    }

    private ParseResult<SqlNegativeValue> Parse_NegativeValue()
    {
        if (TryMatch("-", out var minusSpan))
        {
            if (Try(Parse_SqlIdentifier, out var identifier) && !IsPeekMatch("("))
            {
                return new SqlNegativeValue
                {
                    Span = _text.CreateSpan(minusSpan),
                    Value = identifier.ResultValue,
                };
            }

            var expr = ParseArithmeticExpr();
            if (expr.HasError)
            {
                return expr.Error;
            }

            return new SqlNegativeValue
            {
                Span = _text.CreateSpan(minusSpan),
                Value = expr.ResultValue
            };
        }

        return NoneResult<SqlNegativeValue>();
    }

    private ParseResult<ISqlExpression> Parse_SearchCondition(Func<ParseResult<ISqlExpression>> parseTerm)
    {
        var left = parseTerm();
        while (Try(Parse_LogicalOperator, out var logicalOperator))
        {
            var op = logicalOperator.ResultValue;
            var right = parseTerm();
            left = CreateParseResult(new SqlSearchCondition
            {
                Span = _text.CreateSpan(left.ResultValue.Span, right.ResultValue.Span),
                Left = left.ResultValue,
                LogicalOperator = op.Value,
                OperatorSpan = op.Span,
                Right = right.ResultValue
            }).To<ISqlExpression>();
        }

        return left;
    }

    private ParseResult<List<ISelectColumnExpression>> Parse_SelectColumns()
    {
        var columns = ParseWithComma(() =>
        {
            if (IsPeekKeywords("FROM"))
            {
                return NoneResult<ISelectColumnExpression>();
            }

            var column = Parse_Column_Arithmetic().To<ISelectColumnExpression>();
            if (column.HasError)
            {
                return column.Error;
            }

            var columnExpr = column.ResultValue;
            
            //'Message' + @errorMsg as ErrorMessage
            var visitor = new SqlVisitor();
            var exprList = visitor.Visit(columnExpr);
            if(TryCast<SqlAsExpr>(exprList[^1].Expression, SqlType.AsExpr, out var subAsExpr))
            {
                var previous = exprList[^2].Expression;
                if (TryCast<SqlArithmeticBinaryExpr>(previous, SqlType.ArithmeticBinaryExpr, out var binaryExpr))
                {
                    binaryExpr.Right = subAsExpr.Instance;
                    columnExpr.Alias = subAsExpr.As.ToSql();
                }
            }

            if(TryCast<SqlAsExpr>(columnExpr.Field, SqlType.AsExpr, out var asExpr))
            {
                columnExpr = new SelectColumn()
                {
                    Span = asExpr.Span,
                    Field = asExpr.Instance,
                    Alias = asExpr.As.ToSql()
                };
            }

            if (columnExpr.Field.SqlType == SqlType.ComparisonCondition)
            {
                var condition = (SqlConditionExpression)columnExpr.Field;
                if (condition.ComparisonOperator == ComparisonOperator.Equal)
                {
                    columnExpr = new SelectColumn()
                    {
                        Field = new SqlAssignExpr()
                        {
                            Span = condition.Span,
                            Left = condition.Left,
                            Right = condition.Right
                        }
                    };
                }
            }

            if (!IsPeekKeywords("FROM") && Try(ParseAliasExpr, out var alias))
            {
                columnExpr.Alias = alias.ResultValue.Name;
            }

            if (TryMatch("=", out var equalSpan))
            {
                var rightExpr = ParseArithmeticExpr();
                if (rightExpr.HasError)
                {
                    return rightExpr.Error;
                }

                return new SelectColumn
                {
                    Field = new SqlAssignExpr()
                    {
                        Span = _text.CreateSpan(columnExpr.Span, rightExpr.ResultValue.Span),
                        Left = columnExpr,
                        Right = rightExpr.ResultValue,
                    },
                    Alias = columnExpr.Alias,
                };
            }

            return CreateParseResult(columnExpr);
        });
        return columns;
    }

    private ParseResult<SqlAliasExpr> ParseAliasExpr()
    {
        if (TryKeyword("AS", out _))
        {
            var aliasName = Or(Parse_SqlIdentifierValue, ParseSqlQuotedString)();
            if (aliasName.HasError)
            {
                return aliasName.Error;
            }

            return new SqlAliasExpr()
            {
                Span = aliasName.ResultValue.Span,
                Name = aliasName.ResultValue.Value
            };
        }

        if (Try(Parse_SqlIdentifierNonReservedWord, out var aliasName2))
        {
            return new SqlAliasExpr
            {
                Span = aliasName2.ResultValue.Span,
                Name = aliasName2.ResultValue.Value
            };
        }

        return NoneResult<SqlAliasExpr>();
    }

    private bool TryCast<T>(ISqlExpression expr, SqlType sqlType, out T result)
        where T : ISqlExpression
    {
        if (expr.SqlType == sqlType)
        {
            result = (T)expr;
            return true;
        }
        result = default;
        return false;
    }

    private ParseResult<SqlValue> Parse_SqlIdentifierNonReservedWord()
    {
        if (Try(() => Parse_SqlIdentifierExclude(ReservedWords), out var identifier))
        {
            return identifier;
        }

        return NoneResult<SqlValue>();
    }

    private ParseResult<SqlFieldExpr> Parse_SqlIdentifier()
    {
        if (TryReadSqlIdentifier(out var identifierSpan))
        {
            return new SqlFieldExpr()
            {
                Span = identifierSpan,
                FieldName = identifierSpan.Word
            };
        }

        return NoneResult<SqlFieldExpr>();
    }

    private ParseResult<SqlSetValueStatement> ParseSetValueStatement()
    {
        if (!TryKeywords(["SET"], out var startSpan))
        {
            return NoneResult<SqlSetValueStatement>();
        }

        var name = Parse_SqlIdentifier();
        if (name.Result == null)
        {
            return NoneResult<SqlSetValueStatement>();
        }

        if (!TryMatch("=", out var equalSpan))
        {
            return CreateParseError("Expected =");
        }

        var value = ParseArithmeticExpr().ResultValue;
        return new SqlSetValueStatement()
        {
            Span = _text.CreateSpan(startSpan),
            Name = name.ResultValue,
            Value = value
        };
    }

    private ParseResult<SqlValue> Parse_SqlIdentifierValue()
    {
        if (Try(Parse_SqlIdentifier, out var identifier))
        {
            return new SqlValue()
            {
                Span = identifier.ResultValue.Span,
                Value = identifier.ResultValue.FieldName
            };
        }

        return NoneResult<SqlValue>();
    }

    private ParseResult<SqlValue> Parse_SqlIdentifierExclude(string[] reservedWords)
    {
        var startPosition = _text.Position;
        if (!TryReadSqlIdentifier(out var identifierSpan))
        {
            return NoneResult<SqlValue>();
        }

        if (reservedWords.Contains(identifierSpan.Word.ToUpper()))
        {
            _text.Position = startPosition;
            return NoneResult<SqlValue>();
        }

        return CreateParseResult(new SqlValue()
        {
            Span = identifierSpan,
            Value = identifierSpan.Word
        });
    }


    private ParseResult<SqlTableHintIndex> Parse_TableHintIndex()
    {
        if (!TryKeyword("INDEX", out var startSpan))
        {
            return NoneResult<SqlTableHintIndex>();
        }

        if (TryMatch("=", out var equalSpan))
        {
            var hasParenthesis = TryMatch("(", out var openParenthesis);
            var indexName = _text.ReadSqlIdentifier();
            if (hasParenthesis && !TryMatch(")", out var closeParenthesis))
            {
                return CreateParseError("Expected )");
            }

            if (!hasParenthesis)
            {
                openParenthesis = indexName;
            }

            return new SqlTableHintIndex()
            {
                Span = _text.CreateSpan(openParenthesis),
                IndexValues =
                [
                    new SqlFieldExpr()
                    {
                        Span = indexName,
                        FieldName = indexName.Word
                    }
                ]
            };
        }

        var indexValues = ParseWithoutParenthesesOption(() => ParseWithComma(ParseArithmeticExpr));

        return new SqlTableHintIndex
        {
            Span = _text.CreateSpan(startSpan),
            IndexValues = indexValues.ResultValue
        };
    }

    private bool IsAnyPeekKeyword(params string[] keywords)
    {
        foreach (var keyword in keywords)
        {
            if (IsPeekKeywords(keyword))
            {
                return true;
            }
        }

        return false;
    }

    private ParseResult<ITableSource> Parse_TableSourceWithHints()
    {
        if (!Try(Parse_FromTableSource, out var tableSource))
        {
            return NoneResult<ITableSource>();
        }

        var tableSourceExpr = tableSource.ResultValue;

        if (Try(ParseAliasExpr, out var aliasExpr))
        {
            tableSourceExpr.Alias = aliasExpr.ResultValue.Name;
        }

        if (TryKeyword("WITH", out _))
        {
            MatchSymbol("(");
            var tableHints = ParseWithComma<ISqlExpression>(() =>
            {
                var hintStartPosition = _text.Position;
                if (Try(Parse_TableHintIndex, out var tableHintIndex))
                {
                    return tableHintIndex.ResultValue;
                }

                var hint = ReadSqlIdentifier().Word;
                return new SqlHint()
                {
                    Span = _text.CreateSpan(hintStartPosition),
                    Name = hint
                };
            });
            if (tableHints.HasError)
            {
                return tableHints.Error;
            }

            MatchSymbol(")");
            tableSourceExpr.Withs = tableHints.ResultValue;
        }

        return CreateParseResult(tableSourceExpr);
    }

    private ParseResult<ISqlExpression> Parse_WhereExpression()
    {
        var startPosition = _text.Position;
        var rc = ParseArithmeticExpr();
        if (rc.HasError)
        {
            return rc.Error;
        }

        if (Try(Parse_LogicalOperator, out var logicalOperator))
        {
            var rightExprResult = Parse_WhereExpression();
            if (rightExprResult.HasError)
            {
                return rightExprResult.Error;
            }

            return new SqlSearchCondition
            {
                Span = _text.CreateSpan(startPosition),
                Left = rc.ResultValue,
                LogicalOperator = logicalOperator.Result!.Value,
                Right = rightExprResult.Result
            };
        }

        return rc;
    }

    private ParseResult<ISqlExpression> ParseArithmetic_Primary()
    {
        var startPosition = _text.Position;
        if (Try(Parse_Value_As_DataType, out var value))
        {
            if (value.HasError)
            {
                _text.Position = startPosition;
                return value.Error;
            }

            return value.To<ISqlExpression>();
        }

        if (TryMatch("(", out var openSpan))
        {
            var subExpr = ParseArithmeticExpr();
            if (subExpr.HasError)
            {
                return subExpr.Error;
            }

            if (!TryMatch(")", out var closeSpan))
            {
                return CreateParseError("InvalidOperationException Mismatched parentheses");
            }

            return new SqlGroup
            {
                Span = _text.CreateSpan(startPosition),
                Inner = subExpr.ResultValue
            };
        }

        return CreateParseError("InvalidOperationException Unexpected value");
    }

    private ParseResult<SqlColumnDefinition> ParseColumnConstraints(SqlColumnDefinition sqlColumn)
    {
        if (IsPeekKeywords(","))
        {
            return NoneResult<SqlColumnDefinition>();
        }

        do
        {
            var startPosition = _text.Position;
            if (TryKeywords(["PRIMARY", "KEY"], out _))
            {
                // 最後一個column 有可能沒有逗號 又寫 Table Constraint 的話會被誤判, 所以要檢查是否有 CLUSTERED 
                if (TryKeyword("CLUSTERED", out _))
                {
                    _text.Position = startPosition;
                    break;
                }

                sqlColumn.IsPrimaryKey = true;
                continue;
            }

            if (Try(ParseIdentity, out var identityResult))
            {
                if (identityResult.HasError)
                {
                    return identityResult.Error;
                }

                sqlColumn.Identity = identityResult.ResultValue;
                continue;
            }

            if (Try(ParseDefaultValue, out var defaultValue))
            {
                if (identityResult.HasError)
                {
                    return identityResult.Error;
                }

                sqlColumn.Constraints.Add(defaultValue.ResultValue);
                continue;
            }

            var constraintStartPosition = _text.Position;
            if (TryMatch(ConstraintKeyword, out _))
            {
                var constraintName = _text.ReadSqlIdentifier();
                if (Try(ParseDefaultValue, out var constraintDefaultValue))
                {
                    if (identityResult.HasError)
                    {
                        return identityResult.Error;
                    }

                    var subConstraint = constraintDefaultValue.ResultValue;
                    subConstraint.ConstraintName = constraintName.Word;
                    sqlColumn.Constraints.Add(subConstraint);
                    continue;
                }

                _text.Position = constraintStartPosition;
                var columnConstraint = ParseTableConstraint();
                if (columnConstraint.HasError)
                {
                    return columnConstraint.Error;
                }

                if (columnConstraint.Result == null)
                {
                    return CreateParseError("Expect Constraint DEFAULT");
                }

                sqlColumn.Constraints.Add(columnConstraint.Result);
            }

            if (TryKeywords(["NOT", "FOR", "REPLICATION"], out _))
            {
                sqlColumn.NotForReplication = true;
                continue;
            }

            if (TryKeywords(["NOT", "NULL"], out _))
            {
                sqlColumn.IsNullable = false;
                continue;
            }

            if (TryKeyword("NULL", out _))
            {
                sqlColumn.IsNullable = true;
                continue;
            }

            break;
        } while (true);

        return CreateParseResult(sqlColumn);
    }

    private ParseResult<SqlColumnDefinition> ParseColumnDefinition()
    {
        var startPosition = _text.Position;
        if (!TryReadSqlIdentifier(out var columnNameSpan))
        {
            return NoneResult<SqlColumnDefinition>();
        }

        var columnDefinition = ParseColumnTypeDefinition(columnNameSpan);
        if (columnDefinition.HasError)
        {
            return columnDefinition.Error;
        }

        if (columnDefinition.Result == null)
        {
            _text.Position = startPosition;
            return NoneResult<SqlColumnDefinition>();
        }

        var c = ParseColumnConstraints(columnDefinition.Result);
        if (c.HasError)
        {
            return c.Error;
        }

        return CreateParseResult(columnDefinition.ResultValue);
    }

    private ParseResult<List<SqlConstraintColumn>> ParseColumnsAscDesc()
    {
        var columns = ParseParenthesesWithComma(() =>
        {
            var startPosition = _text.Position;
            var columnName = _text.ReadSqlIdentifier();
            var order = string.Empty;
            if (TryKeyword("ASC", out _))
            {
                order = "ASC";
            }
            else if (TryKeyword("DESC", out _))
            {
                order = "DESC";
            }

            return CreateParseResult(new SqlConstraintColumn
            {
                Span = _text.CreateSpan(startPosition),
                ColumnName = columnName.Word,
                Order = order,
            });
        });
        return columns;
    }

    private ParseResult<SqlColumnDefinition> ParseColumnTypeDefinition(TextSpan columnNameSpan)
    {
        var startPosition = _text.Position;
        var column = new SqlColumnDefinition
        {
            ColumnName = columnNameSpan.Word,
            DataType = ReadSqlIdentifier().Word
        };

        var dataSize = Parse_DataSize();
        if (dataSize.HasError)
        {
            return dataSize.Error;
        }

        column.DataSize = dataSize.Result;
        column.Span = _text.CreateSpan(startPosition);
        return CreateParseResult(column);
    }

    private ParseResult<SqlComputedColumnDefinition> ParseComputedColumnDefinition()
    {
        var startPosition = _text.Position;
        if (!TryReadSqlIdentifier(out var columnNameSpan))
        {
            return NoneResult<SqlComputedColumnDefinition>();
        }

        if (!TryKeyword("AS", out _))
        {
            _text.Position = startPosition;
            return NoneResult<SqlComputedColumnDefinition>();
        }

        if (!TryMatch("(", out _))
        {
            _text.Position = startPosition;
            return CreateParseError("Expected (");
        }

        var computedColumnExpressionSpan = _text.ReadUntilRightParenthesis();

        if (!TryMatch(")", out _))
        {
            _text.Position = startPosition;
            return CreateParseError("Expected )");
        }

        var persist = TryKeyword("PERSISTED", out _);
        var notNull = TryKeywords(["NOT", "NULL"], out _);

        return new SqlComputedColumnDefinition
        {
            Span = _text.CreateSpan(startPosition),
            ColumnName = columnNameSpan.Word,
            Expression = computedColumnExpressionSpan.Word,
            IsPersisted = persist,
            IsNotNull = notNull
        };
    }

    private ParseResult<SqlConstraintDefaultValue> ParseDefaultValue()
    {
        if (!TryKeyword("DEFAULT", out var startSpan))
        {
            return NoneResult<SqlConstraintDefaultValue>();
        }

        TextSpan defaultValue;
        if (TryMatch("(", out _))
        {
            defaultValue = _text.ReadUntilRightParenthesis();
            _text.MatchSymbol(")");
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                Span = _text.CreateSpan(startSpan),
                ConstraintName = string.Empty,
                DefaultValue = defaultValue.Word
            });
        }

        var nullValue = _text.PeekIdentifier("NULL");
        if (nullValue.Length > 0)
        {
            _text.ReadIdentifier();
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                Span = _text.CreateSpan(startSpan),
                ConstraintName = string.Empty,
                DefaultValue = nullValue.Word
            });
        }

        if (_text.Try(_text.ReadSqlIdentifier, out var funcName))
        {
            _text.MatchSymbol("(");
            var funcArgs = _text.ReadUntilRightParenthesis();
            _text.MatchSymbol(")");
            defaultValue = new TextSpan
            {
                Word = $"{funcName.Word}({funcArgs.Word})",
                Offset = funcName.Offset,
                Length = funcName.Length + funcArgs.Length + 2
            };
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                Span = _text.CreateSpan(startSpan),
                ConstraintName = string.Empty,
                DefaultValue = defaultValue.Word,
            });
        }

        if (_text.Try(_text.ReadSqlQuotedString, out var quotedString))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                Span = _text.CreateSpan(startSpan),
                ConstraintName = string.Empty,
                DefaultValue = quotedString.Word,
            });
        }

        if (_text.Try(_text.ReadSqlDate, out var date))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                Span = _text.CreateSpan(startSpan),
                ConstraintName = string.Empty,
                DefaultValue = date.Word,
            });
        }

        if (_text.Try(_text.ReadNegativeNumber, out var negativeNumber))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                Span = _text.CreateSpan(startSpan),
                ConstraintName = string.Empty,
                DefaultValue = negativeNumber.Word,
            });
        }

        if (_text.Try(_text.ReadFloat, out var floatNumber))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                Span = _text.CreateSpan(startSpan),
                ConstraintName = string.Empty,
                DefaultValue = floatNumber.Word,
            });
        }

        defaultValue = _text.ReadInt();
        return CreateParseResult(new SqlConstraintDefaultValue
        {
            Span = _text.CreateSpan(startSpan),
            ConstraintName = string.Empty,
            DefaultValue = defaultValue.Word,
        });
    }

    private ParseResult<SqlValue> ParseFloatValue()
    {
        if (_text.Try(_text.ReadFloat, out var floatNumber))
        {
            return new SqlValue
            {
                Span = floatNumber,
                Value = floatNumber.Word
            };
        }

        return NoneResult<SqlValue>();
    }

    private ParseResult<SqlValue> ParseHexValue()
    {
        if (!TryMatch("0x", out var startSpan))
        {
            return NoneResult<SqlValue>();
        }

        var hexValue = _text.ReadUntil(c => !_text.IsWordChar(c));
        return new SqlValue()
        {
            SqlType = SqlType.HexValue,
            Value = "0x" + hexValue.Word,
            Span = _text.CreateSpan(startSpan),
        };
    }

    private ParseResult<SqlIdentity> ParseIdentity()
    {
        if (!TryMatch("IDENTITY", out var startSpan))
        {
            return NoneResult<SqlIdentity>();
        }

        var sqlIdentity = new SqlIdentity
        {
            Seed = 1,
            Increment = 1
        };
        if (TryMatch("(", out _))
        {
            sqlIdentity.Seed = long.Parse(_text.ReadInt().Word);
            _text.MatchSymbol(",");
            sqlIdentity.Increment = int.Parse(_text.ReadInt().Word);
            _text.MatchSymbol(")");
        }

        sqlIdentity.Span = _text.CreateSpan(startSpan);
        return CreateParseResult(sqlIdentity);
    }

    private ParseResult<SqlValue> ParseIntValue()
    {
        if (_text.Try(_text.ReadInt, out var numberSpan))
        {
            return CreateParseResult(new SqlValue
            {
                SqlType = SqlType.IntValue,
                Value = numberSpan.Word,
                Span = numberSpan
            });
        }

        return NoneResult<SqlValue>();
    }

    private ParseResult<SqlToken> ParseKeywords(params string[] keywords)
    {
        if (TryKeywords(keywords, out var span))
        {
            return CreateParseResult(new SqlToken
            {
                Span = span,
                Value = string.Join(" ", keywords)
            });
        }

        return NoneResult<SqlToken>();
    }

    private ParseResult<SqlToken> Parse_NegativeOrPositive()
    {
        if (TryMatch("-", out var minusSpan))
        {
            return new SqlToken
            {
                Span = minusSpan,
                Value = "-"
            };
        }

        if (TryMatch("+", out var plusSpan))
        {
            return new SqlToken
            {
                Span = plusSpan,
                Value = "+"
            };
        }

        return NoneResult<SqlToken>();
    }

    private ParseResult<SqlValue> ParseNumberValue()
    {
        if (Try(ParseHexValue, out var hexValue))
        {
            if (hexValue.HasError)
            {
                return hexValue.Error;
            }

            return hexValue;
        }

        var startPosition = _text.Position;
        var negativeOrPositive = string.Empty;
        var negativeOrPositiveToken = Parse_NegativeOrPositive().Result;
        if (negativeOrPositiveToken != null)
        {
            negativeOrPositive = negativeOrPositiveToken.Value;
        }


        var number = Or(ParseFloatValue, ParseIntValue)();
        if (number.HasError)
        {
            _text.Position = startPosition;
            return number.Error;
        }

        if (number.Result == null)
        {
            _text.Position = startPosition;
            return NoneResult<SqlValue>();
        }

        var numberExpr = number.ResultValue;
        numberExpr.Span = _text.CreateSpan(startPosition);
        switch (negativeOrPositive)
        {
            case "":
                numberExpr.Value = numberExpr.Value;
                break;
            case "+":
                numberExpr.Value = $"+{numberExpr.Value}";
                break;
            case "-":
                numberExpr.Value = $"-{numberExpr.Value}";
                break;
        }

        return number;
    }

    private ParseResult<SqlOrderByClause> ParseOrderByClause()
    {
        if (!TryKeywords(["ORDER", "BY"], out var startSpan))
        {
            return NoneResult<SqlOrderByClause>();
        }

        var orderByColumns = ParseWithComma<SqlOrderColumn>(() =>
        {
            var columnStartPosition = _text.Position;
            var column = ParseArithmeticExpr().ResultValue;
            var order = OrderType.Asc;
            if (TryKeyword("ASC", out _))
            {
                order = OrderType.Asc;
            }
            else if (TryKeyword("DESC", out _))
            {
                order = OrderType.Desc;
            }

            return new SqlOrderColumn
            {
                Span = _text.CreateSpan(columnStartPosition),
                ColumnName = column,
                Order = order
            };
        });
        if (orderByColumns.HasError)
        {
            return orderByColumns.Error;
        }

        return new SqlOrderByClause
        {
            Span = _text.CreateSpan(startSpan),
            Columns = orderByColumns.ResultValue
        };
    }

    private ParseResult<SqlParameterValue> ParseParameterAssignValue()
    {
        SkipWhiteSpace();
        if (!_text.Try(_text.ReadSqlIdentifier, out var nameSpan))
        {
            return NoneResult<SqlParameterValue>();
        }

        if (!_text.TryMatch("=", out _))
        {
            return CreateParseError("Expected =");
        }

        if (!_text.Try(_text.ReadSqlQuotedString, out var nameValue))
        {
            return CreateParseError($"Expected @name value, but got {_text.PreviousWord().Word}");
        }

        return CreateParseResult(new SqlParameterValue
        {
            Name = nameSpan.Word,
            Value = nameValue.Word
        });
    }

    private ParseResult<SqlParameterValue> ParseParameterValue()
    {
        SkipWhiteSpace();
        var startPosition = _text.Position;
        var valueResult = Parse_Value_As_DataType();
        if (valueResult.HasError)
        {
            _text.Position = startPosition;
            return valueResult.Error;
        }

        if (valueResult.Result == null)
        {
            return NoneResult<SqlParameterValue>();
        }

        if (_text.Peek(_text.ReadSymbols).Word == "=")
        {
            _text.Position = startPosition;
            return NoneResult<SqlParameterValue>();
        }

        return new SqlParameterValue
        {
            Span = _text.CreateSpan(startPosition),
            Name = string.Empty,
            Value = valueResult.ResultValue.ToSql()
        };
    }

    private ParseResult<SqlParameterValue> ParseParameterValueOrAssignValue()
    {
        var rc1 = ParseParameterValue();
        if (rc1.HasError)
        {
            return rc1.Error;
        }

        if (rc1.HasResult && rc1.Result != null)
        {
            return rc1;
        }

        var rc2 = ParseParameterAssignValue();
        if (rc2.HasError)
        {
            return rc2.Error;
        }

        if (rc2.HasResult && rc2.Result != null)
        {
            return rc2;
        }

        return NoneResult<SqlParameterValue>();
    }


    private ParseResult<T> ParseWithParentheses<T>(Func<ParseResult<T>> parseElemFn)
    {
        if (!TryMatch("(", out _))
        {
            return CreateParseError("Expected (");
        }

        var inner = parseElemFn();
        if (inner.HasError)
        {
            return inner.Error;
        }

        if (!TryMatch(")", out _))
        {
            return CreateParseError("Expected )");
        }

        return inner;
    }

    private ParseResult<SqlGroup> ParseParenthesesWith<T>(Func<ParseResult<T>> parseElemFn)
        where T : ISqlExpression
    {
        if (!TryMatch("(", out var startSpan))
        {
            return CreateParseError("Expected (");
        }

        var inner = parseElemFn();
        if (inner.HasError)
        {
            return inner.Error;
        }

        if (!TryMatch(")", out var endSpan))
        {
            return CreateParseError("Expected )");
        }

        return new SqlGroup
        {
            Span = _text.CreateSpan(startSpan, endSpan),
            Inner = inner.ResultValue
        };
    }

    private ParseResult<ISqlExpression> ParseParenthesesOption<T>(Func<ParseResult<T>> parseElemFn)
        where T : ISqlExpression
    {
        var has = TryMatch("(", out var startSpan);
        var inner = parseElemFn();
        if (inner.HasError)
        {
            return inner.Error;
        }

        var endSpan = new TextSpan();
        if (has && !TryMatch(")", out endSpan))
        {
            return CreateParseError("Expected )");
        }

        if (has)
        {
            return new SqlGroup()
            {
                Span = _text.CreateSpan(startSpan, endSpan),
                Inner = inner.ResultValue
            };
        }

        return inner.To<ISqlExpression>();
    }

    private ParseResult<T> ParseWithoutParenthesesOption<T>(Func<ParseResult<T>> parseElemFn)
    {
        var has = TryMatch("(", out var startSpan);
        var inner = parseElemFn();
        if (inner.HasError)
        {
            return inner.Error;
        }

        var endSpan = new TextSpan();
        if (has && !TryMatch(")", out endSpan))
        {
            return CreateParseError("Expected )");
        }

        if (has && inner.Result is ISqlExpression sqlExpr)
        {
            sqlExpr.Span = _text.CreateSpan(startSpan, endSpan);
        }

        return inner;
    }

    private ParseResult<List<T>> ParseParenthesesWithComma<T>(Func<ParseResult<T>> parseElemFn)
    {
        if (!TryMatch("(", out _))
        {
            return CreateParseError("Expected (");
        }

        var elements = ParseWithComma(parseElemFn);
        if (elements.HasError)
        {
            return elements.Error;
        }

        if (!TryMatch(")", out _))
        {
            return CreateParseError("Expected )");
        }

        return elements;
    }

    private ParseResult<SqlConstraintPrimaryKeyOrUnique> ParsePrimaryKeyOrUnique()
    {
        var sqlConstraint = new SqlConstraintPrimaryKeyOrUnique();
        var primaryKeyOrUniqueToken = GetResult(Or(Keywords("PRIMARY", "KEY"), Keywords("UNIQUE")));
        if (primaryKeyOrUniqueToken != null)
        {
            sqlConstraint.ConstraintType = primaryKeyOrUniqueToken.Value;
        }

        if (string.IsNullOrEmpty(sqlConstraint.ConstraintType))
        {
            return NoneResult<SqlConstraintPrimaryKeyOrUnique>();
        }

        var clusteredToken = GetResult(Or(Keywords("CLUSTERED"), Keywords("NONCLUSTERED")));
        if (clusteredToken != null)
        {
            sqlConstraint.Clustered = clusteredToken.Value;
        }

        var columnsSpan = ParseColumnsAscDesc();
        if (columnsSpan.HasError)
        {
            return columnsSpan.Error;
        }

        sqlConstraint.Columns = columnsSpan.ResultValue;
        return sqlConstraint;
    }

    private ParseResult<SqlConstraintPrimaryKeyOrUnique> ParsePrimaryKeyOrUniqueExpression()
    {
        var primaryKeyOrUniqueSpan = ParsePrimaryKeyOrUnique();
        if (primaryKeyOrUniqueSpan.HasError)
        {
            return primaryKeyOrUniqueSpan.Error;
        }

        if (primaryKeyOrUniqueSpan.Result == null)
        {
            return NoneResult<SqlConstraintPrimaryKeyOrUnique>();
        }

        var sqlConstraint = primaryKeyOrUniqueSpan.ResultValue;
        if (TryKeyword("WITH", out _))
        {
            var togglesSpan = ParseParenthesesWithComma(ParseWithToggle);
            if (togglesSpan.HasError)
            {
                return togglesSpan.Error;
            }

            sqlConstraint.WithToggles = togglesSpan.ResultValue;
        }

        if (TryKeyword("ON", out _))
        {
            sqlConstraint.On = ReadSqlIdentifier().Word;
        }

        if (Try(ParseIdentity, out var identitySpan))
        {
            if (identitySpan.HasError)
            {
                return identitySpan.Error;
            }

            sqlConstraint.Identity = identitySpan.ResultValue;
        }

        return sqlConstraint;
    }

    private ParseResult<ReferentialAction> ParseReferentialAction()
    {
        var actionTokenSpan = One(Keywords("NO", "ACTION"), Keywords("CASCADE"), Keywords("SET", "NULL"),
            Keywords("SET", "DEFAULT"))();
        if (actionTokenSpan.HasError)
        {
            return actionTokenSpan.Error;
        }

        var actionToken = actionTokenSpan.ResultValue;
        var action = actionToken.Value.ToUpper() switch
        {
            "NO ACTION" => ReferentialAction.NoAction,
            "CASCADE" => ReferentialAction.Cascade,
            "SET NULL" => ReferentialAction.SetNull,
            "SET DEFAULT" => ReferentialAction.SetDefault,
            _ => ReferentialAction.NoAction
        };
        return new ParseResult<ReferentialAction>(action);
    }

    private ParseResult<SqlValue> ParseSqlQuotedString()
    {
        if (_text.Try(_text.ReadSqlQuotedString, out var quotedStringSpan))
        {
            return new SqlValue
            {
                Span = quotedStringSpan,
                Value = quotedStringSpan.Word,
            };
        }

        return NoneResult<SqlValue>();
    }

    private ParseResult<ISqlConstraint> ParseTableConstraint()
    {
        var constraintName = string.Empty;
        if (TryKeyword(ConstraintKeyword, out _))
        {
            constraintName = ReadSqlIdentifier().Word;
        }

        var tablePrimaryKeyOrUniqueExpr = ParsePrimaryKeyOrUniqueExpression();
        if (tablePrimaryKeyOrUniqueExpr.HasError)
        {
            return tablePrimaryKeyOrUniqueExpr.Error;
        }

        if (tablePrimaryKeyOrUniqueExpr.Result != null)
        {
            tablePrimaryKeyOrUniqueExpr.Result.ConstraintName = constraintName;
            return tablePrimaryKeyOrUniqueExpr.Result;
        }

        var tableForeignKeyExpr = ParseForeignKeyExpression();
        if (tableForeignKeyExpr.HasError)
        {
            return tableForeignKeyExpr.Error;
        }

        if (tableForeignKeyExpr.Result != null)
        {
            tableForeignKeyExpr.Result.ConstraintName = constraintName;
            return tableForeignKeyExpr.Result;
        }

        return NoneResult<ISqlConstraint>();
    }

    private ParseResult<SqlFieldExpr> ParseTableName()
    {
        if (_text.Try(_text.ReadIdentifier, out var fieldNameSpan))
        {
            return CreateParseResult(new SqlFieldExpr()
            {
                Span = fieldNameSpan,
                FieldName = fieldNameSpan.Word
            });
        }

        return NoneResult<SqlFieldExpr>();
    }

    private ParseResult<SqlValues> Parse_Values()
    {
        var startPosition = _text.Position;
        if (!TryMatch("(", out var startSpan))
        {
            return NoneResult<SqlValues>();
        }

        var items = ParseWithComma(() =>
        {
            var value = ParseArithmeticExpr();
            if (value.HasError)
            {
                return value.Error;
            }

            return value;
        });
        if (items.HasError)
        {
            _text.Position = startPosition;
            return items.Error;
        }

        if (!TryMatch(")", out _))
        {
            _text.Position = startPosition;
            return NoneResult<SqlValues>();
        }

        if (items.ResultValue.Count <= 1)
        {
            _text.Position = startPosition;
            return NoneResult<SqlValues>();
        }

        return new SqlValues
        {
            Span = _text.CreateSpan(startSpan),
            Items = items.ResultValue.ToList()
        };
    }

    private ParseResult<List<T>> ParseWithComma<T>(Func<ParseResult<T>> parseElemFn)
    {
        var elements = new List<T>();
        do
        {
            if (PeekBracket().Equals(")"))
            {
                break;
            }

            var elem = parseElemFn();
            if (elem is { HasResult: true, Result: null })
            {
                break;
            }

            if (elem.HasError)
            {
                return elem.Error;
            }

            elements.Add(elem.ResultValue);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();
        } while (!_text.IsEnd());

        return CreateParseResult(elements);
    }

    private ParseResult<SqlToggle> ParseWithToggle()
    {
        var startPosition = _text.Position;
        var toggleName = _text.ReadSqlIdentifier();
        if (toggleName.Length == 0)
        {
            _text.Position = startPosition;
            return NoneResult<SqlToggle>();
        }

        var toggle = new SqlToggle
        {
            ToggleName = toggleName.Word
        };

        if (!_text.TryMatch("=", out _))
        {
            _text.Position = startPosition;
            return CreateParseError("Expected toggleName =");
        }

        if (_text.Try(_text.ReadInt, out var number))
        {
            toggle.Value = number.Word;
            toggle.Span = _text.CreateSpan(startPosition);
            return CreateParseResult(toggle);
        }

        toggle.Value = _text.ReadSqlIdentifier().Word;
        toggle.Span = _text.CreateSpan(startPosition);
        return CreateParseResult(toggle);
    }

    private string PeekBracket()
    {
        SkipWhiteSpace();
        return _text.Peek(() => _text.ReadBracket()).Word;
    }

    private Func<ParseResult<SqlToken>> PeekKeywords(params string[] keywords)
    {
        return () =>
        {
            var startPosition = _text.Position;
            var result = ParseKeywords(keywords);
            _text.Position = startPosition;
            return result;
        };
    }

    private string PeekSymbolString(int length)
    {
        SkipWhiteSpace();
        return _text.Peek(() => _text.ReadSymbol(length)).Word;
    }

    private bool TryPeekSymbolContains(int length, string[] symbols, out string actual)
    {
        actual = PeekSymbolString(length);
        if (symbols.Contains(actual))
        {
            return true;
        }

        actual = string.Empty;
        return false;
    }

    private TextSpan ReadSqlIdentifier()
    {
        SkipWhiteSpace();
        return _text.ReadSqlIdentifier();
    }

    private string ReadSymbolString(int length)
    {
        SkipWhiteSpace();
        var span = _text.NextText(length);
        return span.Word;
    }

    private void SkipWhiteSpace()
    {
        while (true)
        {
            var isSkip1 = _text.SkipWhitespace();
            var isSkip2 = _text.SkipSqlComment();
            var isSkip = isSkip1 || isSkip2;
            if (!isSkip)
            {
                break;
            }
        }
    }

    private Func<ParseResult<SqlToken>> Symbol(string symbol)
    {
        return () =>
        {
            SkipWhiteSpace();
            if (_text.TryMatch(symbol, out var symbolSpan))
            {
                return new SqlToken
                {
                    Span = symbolSpan,
                    Value = symbol
                };
            }

            return NoneResult<SqlToken>();
        };
    }

    private Func<ParseResult<SqlToken>> SymbolWithNoncontinuous(string symbol)
    {
        return () =>
        {
            var startPosition = _text.Position;
            foreach (var symbolChar in symbol)
            {
                SkipWhiteSpace();
                var currentChar = _text.ReadChar();
                if (currentChar != symbolChar)
                {
                    _text.Position = startPosition;
                    return NoneResult<SqlToken>();
                }
            }

            return new SqlToken
            {
                Span = new TextSpan()
                {
                    Offset = startPosition,
                    Length = _text.Position - startPosition
                },
                Value = symbol.Replace(" ", "").Replace("\t", "")
            };
        };
    }

    private bool TryKeyword(string expected, out TextSpan textSpan)
    {
        SkipWhiteSpace();
        return _text.TryKeywordIgnoreCase(expected, out textSpan);
    }

    private bool TryKeywords(string[] keywords, out TextSpan span)
    {
        SkipWhiteSpace();
        return _text.TryKeywordsIgnoreCase(keywords, out span);
    }

    private bool TryMatch(string expected, out TextSpan textSpan)
    {
        SkipWhiteSpace();
        return _text.TryMatch(expected, out textSpan);
    }

    private ParseResult<SqlValue> Parse_QuotedString()
    {
        SkipWhiteSpace();
        var token = _text.ReadSqlQuotedString();
        if (token.Length == 0)
        {
            return NoneResult<SqlValue>();
        }

        return new SqlValue
        {
            Span = token,
            Value = token.Word
        };
    }

    private bool TryReadSqlIdentifier(out TextSpan result)
    {
        SkipWhiteSpace();
        if (!_text.Try(_text.ReadSqlIdentifier, out result))
        {
            return false;
        }

        return true;
    }

    private bool TryReadSqlFunctionName(out TextSpan result)
    {
        SkipWhiteSpace();
        if (!_text.Try(_text.ReadSqlIdentifier, out result))
        {
            return false;
        }

        return true;
    }
}

public class SqlAsExprVisitor : SqlVisitor
{
    public bool HasAs { get; set; }
    
}