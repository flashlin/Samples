using System.Diagnostics;
using System.Runtime.InteropServices.JavaScript;
using System.Text.RegularExpressions;
using T1.SqlSharp;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlParser
{
    private const string ConstraintKeyword = "CONSTRAINT";
    private static readonly string[] ReservedWords = ["FROM", "SELECT", "JOIN", "LEFT", "UNION", "ON", "GROUP", "WITH", "WHERE", "UNPIVOT", "FOR"];

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

    public static ParseResult<ISqlExpression> Parse(string sql)
    {
        var p = new SqlParser(sql);
        return p.Parse();
    }

    public ParseResult<ISqlExpression> Parse()
    {
        if (Try(ParseCreateTableStatement, out var createTableResult))
        {
            return createTableResult.Result;
        }

        if (Try(ParseSelectStatement, out var selectResult))
        {
            return selectResult.Result;
        }

        if (Try(ParseExecSpAddExtendedProperty, out var execSpAddExtendedPropertyResult))
        {
            return execSpAddExtendedPropertyResult.Result;
        }

        return CreateParseError("Unknown statement");
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

    private ParseResult<SqlGroupByClause> ParseGroupByClause()
    {
        if (!TryKeywords(["GROUP", "BY"], out _))
        {
            return NoneResult<SqlGroupByClause>();
        }

        var groupByColumns = ParseWithComma(ParseArithmeticExpr);
        if (groupByColumns.HasError)
        {
            return groupByColumns.Error;
        }

        return CreateParseResult(new SqlGroupByClause
        {
            Columns = groupByColumns.ResultValue
        });
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

        topClause.Span = new TextSpan()
        {
            Word =  _text.GetText(startSpan.Offset, _text.Position),
            Offset = startSpan.Offset,
            Length = _text.Position - startSpan.Offset,
        };
        return CreateParseResult(topClause);
    }

    public ParseResult<ISqlExpression> ParseValue()
    {
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
                Span = TextSpan.FromBound(openParenthesis, closeParenthesis),
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

        if (TryKeyword("DISTINCT", out _))
        {
            var value = ParseArithmeticExpr();
            if (value.HasError)
            {
                return value.Error;
            }

            return new SqlDistinct()
            {
                Value = value.ResultValue
            };
        }

        if (TryKeyword("NULL", out _))
        {
            return new SqlNullValue();
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

        if (TryReadSqlIdentifier(out var identifier))
        {
            return new SqlFieldExpr
            {
                FieldName = identifier.Word
            };
        }

        if (Try(ParseTableName, out var tableName))
        {
            return tableName.ResultValue;
        }

        return NoneResult<ISqlExpression>();
    }

    public ParseResult<ISqlExpression> Parse_Value_As_DataType()
    {
        var valueExpr = ParseValue();
        if (valueExpr.HasError)
        {
            return valueExpr.Error;
        }

        if (valueExpr.Result == null)
        {
            return NoneResult<ISqlExpression>();
        }

        if (TryKeyword("AS", out _))
        {
            var dataType = Or<ISqlExpression>(Parse_DataType, ParseSqlQuotedString)();
            return new SqlAsExpr
            {
                Instance = valueExpr.ResultValue,
                As = dataType.ResultValue
            };
        }

        return valueExpr;
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
        while (PeekSymbolString(1).Equals("*") || PeekSymbolString(1).Equals("/"))
        {
            var op = ReadSymbolString(1);
            var right = parseTerm();
            left = new SqlArithmeticBinaryExpr
            {
                Left = left.ResultValue,
                Operator = op.ToArithmeticOperator(),
                Right = right.ResultValue,
            };
        }

        return left;
    }

    public ParseResult<ISqlExpression> ParseArithmeticExpr()
    {
        return
            Parse_SearchCondition(
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
        if (!TryKeywords(["CREATE", "TABLE"], out _))
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

        return CreateParseResult(createTableStatement);
    }

    public ParseResult<SqlSpAddExtendedPropertyExpression> ParseExecSpAddExtendedProperty()
    {
        if (!TryKeywords(["EXEC", "SP_AddExtendedProperty"], out _))
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
        if (!TryKeywords(["FOREIGN", "KEY"], out _))
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
        if (!TryKeyword("SELECT", out _))
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
            var tableSources = Parse_FromTableSources();
            if (tableSources.HasError)
            {
                return tableSources.Error;
            }
            selectStatement.FromSources = tableSources.ResultValue;
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
        
        if(Try(ParseForXmlClause, out var forXmlClause))
        {
            selectStatement.ForXml = forXmlClause.ResultValue;
        }

        if(Try(ParseUnionSelectClauseList, out var unionSelectClauseList))
        {
            selectStatement.Unions = unionSelectClauseList.ResultValue;
        }

        SkipStatementEnd();
        return CreateParseResult(selectStatement);
    }

    private ParseResult<ISqlForXmlClause> ParseForXmlClause()
    {
        if(Try(ParseForXmlPathClause, out var forXmlPathClause))
        {
            return forXmlPathClause.ResultValue;
        }
        if(Try(ParseForXmlAutoClause, out var forXmlAutoClause))
        {
            return forXmlAutoClause.ResultValue;
        }
        return NoneResult<ISqlForXmlClause>();
    }
    
    private ParseResult<SqlUnpivotClause> ParseUnpivotClause()
    {
        if (!TryKeyword("UNPIVOT", out _))
        {
            return NoneResult<SqlUnpivotClause>();
        }
        if(!TryMatch("(", out var openParenthesis))
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
        
        if(!TryMatch(")", out var closeParenthesis))
        {
            return CreateParseError("Expected )");
        }

        var alias = ParseAliasExpr();
        
        return CreateParseResult(new SqlUnpivotClause
        {
            NewColumn = newColumn.ResultValue,
            ForSource = forSource.ResultValue,
            InColumns = inColumns.ResultValue,
            AliasName = alias.ResultValue.Name  
        });
    }
    
    private ParseResult<SqlForXmlPathClause> ParseForXmlPathClause()
    {
        if (!TryKeywords(["FOR", "XML", "PATH"], out _))
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
        return forXmlClause;
    }
    
    private ParseResult<SqlForXmlAutoClause> ParseForXmlAutoClause()
    {
        if (!TryKeywords(["FOR", "XML", "AUTO"], out _))
        {
            return NoneResult<SqlForXmlAutoClause>();
        }
        var forXmlClause = new SqlForXmlAutoClause();
        forXmlClause.CommonDirectives = Parse_ForXmlRootDirectives();
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
            if (TryKeyword("ROOT", out _))
            {
                var rootName = ParseWithParentheses(ParseValue);
                return new SqlForXmlRootDirective
                {
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
        if (!TryKeywords(["UNION", "ALL"], out _))
        {
            if (!TryKeyword("UNION", out _))
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
                Span = TextSpan.FromBound(openSpan, closeSpan),
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
                Span = TextSpan.FromBound(startSpan, expr.ResultValue.Span),
                Operator = UnaryOperator.BitwiseNot,
                Operand = expr.ResultValue
            };
        }

        return NoneResult<SqlUnaryExpr>();
    }

    private ParseResult<SqlRankClause> ParseRankClause()
    {
        var startPosition = _text.Position;
        if (!TryKeyword("RANK", out _))
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
            PartitionBy = partitionBy,
            OrderBy = orderBy
        };
    }

    private ParseResult<SqlPartitionByClause> ParsePartitionBy()
    {
        if (!TryKeywords(["PARTITION", "BY"], out _))
        {
            return NoneResult<SqlPartitionByClause>();
        }

        var columns = ParseWithComma(ParseValue);
        if (columns.HasError)
        {
            return columns.Error;
        }

        return CreateParseResult(new SqlPartitionByClause
        {
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
                function.Parameters[0] = new SqlDataType
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
            if(searchCondition.LogicalOperator == LogicalOperator.And)
            {
                // return new SqlBetweenValue
                // {
                //     Start = searchCondition.Left,
                //     End = searchCondition.Right
                // };
            }
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
            Start = start.ResultValue,
            End = end.ResultValue
        };
    }

    private ParseResult<SqlWhenThenClause> Parse_Case_WhenClause()
    {
        if (!TryKeyword("WHEN", out _))
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
            When = whenExpr.ResultValue,
            Then = thenExpr.ResultValue
        });
    }

    private ParseResult<SqlCaseCaluse> ParseCaseClause()
    {
        if (!TryKeyword("CASE", out _))
        {
            return NoneResult<SqlCaseCaluse>();
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

        return CreateParseResult(new SqlCaseCaluse
        {
            When = whenExpr,
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
                Field = arithmetic.ResultValue
            });
        }

        return NoneResult<SelectColumn>();
    }

    private ParseResult<ComparisonOperator?> Parse_ComparisonOperator()
    {
        var rc = Or(
            Keywords("IS", "NOT"),
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
            return NoneResult<ComparisonOperator?>();
        }

        return rc.Result.Value.ToUpper().ToComparisonOperator();
    }

    private ParseResult<ISqlExpression> Parse_ConditionExpr(Func<ParseResult<ISqlExpression>> parseTerm)
    {
        var left = parseTerm();
        while (Try(Parse_ComparisonOperator, out var comparisonOperator))
        {
            var op = comparisonOperator.Result!.Value;

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
                Left = left.ResultValue,
                ComparisonOperator = op,
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

        return dataSize;
    }

    private ParseResult<SqlDataType> Parse_DataType()
    {
        if (!TryReadSqlIdentifier(out var identifier))
        {
            return NoneResult<SqlDataType>();
        }

        var dataType = Parse_DataSize();
        if (dataType.HasError)
        {
            return dataType.Error;
        }

        return new SqlDataType()
        {
            DataTypeName = identifier.Word,
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

        if (Try(ParseFunctionCall, out var function))
        {
            return new SqlFuncTableSource()
            {
                Function = function.ResultValue
            };
        }

        if (Try(Parse_SqlIdentifier, out var tableName))
        {
            return new SqlTableSource()
            {
                TableName = tableName.ResultValue.Value
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
                    FunctionName = identifier.Word,
                    Parameters = parameters.ResultValue!.ToArray()
                };
                function = NormalizeFunctionName(function);
                return function;
            }
        }
        _text.Position = startPosition;
        return NoneResult<SqlFunctionExpression>();
    }

    private ParseResult<SqlJoinTableCondition> Parse_JoinTableSource()
    {
        if (TryKeywords(["INNER", "JOIN"], out _))
        {
            var tableSource = Parse_JoinTableSourceOn();
            return tableSource.ResultValue;
        }

        if (TryKeyword("JOIN", out _))
        {
            var tableSource = Parse_JoinTableSourceOn();
            return tableSource.ResultValue;
        }

        if (TryKeywords(["LEFT", "JOIN"], out _))
        {
            var tableSource = Parse_JoinTableSourceOn().ResultValue;
            tableSource.JoinType = JoinType.Left;
            return tableSource;
        }

        if (TryKeywords(["RIGHT", "JOIN"], out _))
        {
            var tableSource = Parse_JoinTableSourceOn().ResultValue;
            tableSource.JoinType = JoinType.Right;
            return tableSource;
        }

        return NoneResult<SqlJoinTableCondition>();
    }

    private ParseResult<SqlJoinTableCondition> Parse_JoinTableSourceOn()
    {
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
            JoinedTable = tableSource.ResultValue,
            OnCondition = onCondition.ResultValue,
        };
    }

    private ParseResult<LogicalOperator?> Parse_LogicalOperator()
    {
        var rc = Or(Keywords("AND"), Keywords("OR"), Keywords("NOT"))();
        if (rc.HasError)
        {
            return rc.Error;
        }

        if (rc.Result == null)
        {
            return NoneResult<LogicalOperator?>();
        }

        var logicalOperator = rc.Result.Value.ToUpper() switch
        {
            "AND" => LogicalOperator.And,
            "OR" => LogicalOperator.Or,
            "NOT" => LogicalOperator.Not,
            _ => LogicalOperator.None
        };
        return logicalOperator;
    }

    private ParseResult<SqlNegativeValue> Parse_NegativeValue()
    {
        if (TryMatch("-", out var minusSpan))
        {
            if (Try(Parse_SqlIdentifier, out var identifier) && !IsPeekMatch("("))
            {
                return new SqlNegativeValue
                {
                    Value = new SqlFieldExpr()
                    {
                        FieldName = identifier.ResultValue.Value
                    }
                };
            }

            var expr = ParseArithmeticExpr();
            if (expr.HasError)
            {
                return expr.Error;
            }

            return new SqlNegativeValue
            {
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
            var op = logicalOperator.Result!.Value;
            var right = parseTerm();
            left = CreateParseResult(new SqlSearchCondition
            {
                Left = left.ResultValue,
                LogicalOperator = op,
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

            if (columnExpr.Field.SqlType == SqlType.AsExpr)
            {
                var asExpr = (SqlAsExpr)columnExpr.Field;
                columnExpr = new SelectColumn()
                {
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
            var aliasName = Or(Parse_SqlIdentifier, ParseSqlQuotedString)();
            if (aliasName.HasError)
            {
                return aliasName.Error;
            }

            return new SqlAliasExpr()
            {
                Name = aliasName.ResultValue.Value
            };
        }

        if (Try(Parse_SqlIdentifierNonReservedWord, out var aliasName2))
        {
            return new SqlAliasExpr
            {
                Name = aliasName2.ResultValue.Value
            };
        }

        return NoneResult<SqlAliasExpr>();
    }
    
    private ParseResult<SqlValue> Parse_SqlIdentifierNonReservedWord()
    {
        if (Try(() => Parse_SqlIdentifierExclude(ReservedWords), out var identifier))
        {
            return identifier;
        }
        return NoneResult<SqlValue>();
    }

    private ParseResult<SqlValue> Parse_SqlIdentifier()
    {
        if (TryReadSqlIdentifier(out var identifierSpan))
        {
            return new SqlValue
            {
                Span = identifierSpan,
                Value = identifierSpan.Word
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
        if (!TryKeyword("INDEX", out _))
        {
            return NoneResult<SqlTableHintIndex>();
        }

        if (TryMatch("=", out var equalSpan))
        {
            if (!TryMatch("(", out var openParenthesis))
            {
                return CreateParseError("Expected (");
            }

            var indexName = _text.ReadSqlIdentifier();
            if (!TryMatch(")", out var closeParenthesis))
            {
                return CreateParseError("Expected )");
            }

            return new SqlTableHintIndex()
            {
                IndexValues = [indexName.Word]
            };
        }

        if (!TryMatch("(", out _)) 
        {
            return CreateParseError("Expected (");
        }

        var indexValues = ParseWithComma<string>(() =>
        {
            var indexName = _text.ReadSqlIdentifier();
            return indexName.Word;
        });

        if (!TryMatch(")", out _)) 
        {
            return CreateParseError("Expected )");
        }

        return new SqlTableHintIndex
        {
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
                if (Try(Parse_TableHintIndex, out var tableHintIndex))
                {
                    return tableHintIndex.ResultValue;
                }

                var hint = ReadSqlIdentifier().Word;
                return new SqlHint()
                {
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
        //var rc = Or<ISqlExpression>(Parse_SearchCondition, Parse_ConditionExpression)();
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
                Span = new TextSpan
                {
                    Offset = openSpan.Offset,
                    Length = closeSpan.Offset - openSpan.Offset + closeSpan.Length
                },
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
                ColumnName = columnName.Word,
                Order = order,
            });
        });
        return columns;
    }

    private ParseResult<SqlColumnDefinition> ParseColumnTypeDefinition(TextSpan columnNameSpan)
    {
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
        return CreateParseResult(column);
    }

    /*
     * <computed_column_definition> ::=
column_name AS computed_column_expression
[ PERSISTED [ NOT NULL ] ]
[
    [ CONSTRAINT constraint_name ]
    { PRIMARY KEY | UNIQUE }
        [ CLUSTERED | NONCLUSTERED ]
        [
            WITH FILLFACTOR = fillfactor
          | WITH ( <index_option> [ ,... n ] )
        ]
        [ ON { partition_scheme_name ( partition_column_name )
        | filegroup | "default" } ]

    | [ FOREIGN KEY ]
        REFERENCES referenced_table_name [ ( ref_column ) ]
        [ ON DELETE { NO ACTION | CASCADE } ]
        [ ON UPDATE { NO ACTION } ]
        [ NOT FOR REPLICATION ]

    | CHECK [ NOT FOR REPLICATION ] ( logical_expression )
]
     */
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
        var notNull = TryKeywords(["NOT", "NULL"], out  _);

        return CreateParseResult(new SqlComputedColumnDefinition
        {
            ColumnName = columnNameSpan.Word,
            Expression = computedColumnExpressionSpan.Word,
            IsPersisted = persist,
            IsNotNull = notNull
        });
    }

    private ParseResult<SqlConstraintDefaultValue> ParseDefaultValue()
    {
        if (!TryKeyword("DEFAULT", out _))
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
                ConstraintName = string.Empty,
                DefaultValue = defaultValue.Word,
            });
        }

        if (_text.Try(_text.ReadSqlQuotedString, out var quotedString))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = quotedString.Word,
            });
        }

        if (_text.Try(_text.ReadSqlDate, out var date))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = date.Word,
            });
        }

        if (_text.Try(_text.ReadNegativeNumber, out var negativeNumber))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = negativeNumber.Word,
            });
        }

        if (_text.Try(_text.ReadFloat, out var floatNumber))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = floatNumber.Word,
            });
        }

        defaultValue = _text.ReadInt();
        return CreateParseResult(new SqlConstraintDefaultValue
        {
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
            Span = new TextSpan
            {
                Offset = startSpan.Offset,
                Length = hexValue.Offset - startSpan.Offset + hexValue.Length
            }
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

        return CreateParseResult(sqlIdentity);
    }

    private ParseResult<SqlValue> ParseIntValue()
    {
        if (_text.Try(_text.ReadInt, out var number))
        {
            return CreateParseResult(new SqlValue
            {
                SqlType = SqlType.IntValue,
                Value = number.Word,
                Span = number
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
        var negative = TryMatch("-", out _);
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

        number.ResultValue.Value = negative ? $"-{number.ResultValue.Value}" : number.ResultValue.Value;
        return number;
    }

    private ParseResult<SqlOrderByClause> ParseOrderByClause()
    {
        if (!TryKeywords(["ORDER", "BY"], out _))
        {
            return NoneResult<SqlOrderByClause>();
        }

        var orderByColumns = ParseWithComma<SqlOrderColumn>(() =>
        {
            var column = ReadSqlIdentifier().Word;
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
            Columns = orderByColumns.ResultValue
        };
    }

    private ParseResult<SqlParameterValue> ParseParameterAssignValue()
    {
        SkipWhiteSpace();
        if (!_text.Try(_text.ReadSqlIdentifier, out var name))
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
            Name = name.Word,
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
        if (_text.Try(_text.ReadSqlQuotedString, out var quotedString))
        {
            return new SqlValue
            {
                Value = quotedString.Word,
                Span = quotedString
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
        if (_text.Try(_text.ReadIdentifier, out var fieldName))
        {
            return CreateParseResult(new SqlFieldExpr()
            {
                FieldName = fieldName.Word
            });
        }

        return NoneResult<SqlFieldExpr>();
    }

    private ParseResult<SqlValues> Parse_Values()
    {
        var startPosition = _text.Position;
        if (!TryMatch("(", out _))
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
            return CreateParseResult(toggle);
        }

        toggle.Value = _text.ReadSqlIdentifier().Word;
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
        var startPosition = _text.Position;
        if (!_text.Try(_text.ReadSqlIdentifier, out result))
        {
            return false;
        }

        // if (ReservedWords.Contains(result.Word.ToUpper()))
        // {
        //     _text.Position = startPosition;
        //     result = new TextSpan
        //     {
        //         Word = string.Empty,
        //         Offset = startPosition,
        //         Length = 0
        //     };
        //     return false;
        // }

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