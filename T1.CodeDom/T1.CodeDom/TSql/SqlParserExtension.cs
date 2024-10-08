﻿using System;
using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.CodeDom.TSql.Parselets;
using T1.Standard.IO;

namespace T1.CodeDom.TSql
{
    public static class SqlParserExtension
    {
        public delegate bool TryConsumeDelegate(IParser scanner, out SqlCodeExpr expr);

        public static SqlCodeExpr Consume(this IParser parser, SqlToken tokenType)
        {
            if (!parser.TryConsumeAny(0, out var expr, tokenType))
            {
                ThrowHelper.ThrowParseException(parser, $"Expect '{tokenType}'.");
            }

            return expr;
        }

        public static SqlCodeExpr ConsumeAny(this IParser parser, params Func<IParser, SqlCodeExpr>[] parseFuncList)
        {
            var expr = parser.ParseAny(parseFuncList);
            if (expr == null)
            {
                var helpMessage = parser.Scanner.GetHelpMessage();
                throw new ParseException(helpMessage);
            }

            return expr;
        }

        public static List<ArgumentSqlCodeExpr> ConsumeArgumentList(this IParser parser)
        {
            var arguments = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
            {
                var comments = parser.IgnoreComments();

                if (!parser.TryConsume(SqlToken.Variable, out var varName))
                {
                    return null;
                }

                parser.Scanner.Match(SqlToken.As);

                var dataType = parser.ConsumeDataType();

                SqlCodeExpr defaultValueExpr = null;
                if (parser.Scanner.Match(SqlToken.Equal))
                {
                    defaultValueExpr = parser.ParseExp() as SqlCodeExpr;
                }

                var isOutput = false;
                if (parser.Scanner.Match(SqlToken.Output))
                {
                    isOutput = true;
                }

                return new ArgumentSqlCodeExpr
                {
                    Comments = comments,
                    Name = varName as SqlCodeExpr,
                    DataType = dataType,
                    IsOutput = isOutput,
                    DefaultValueExpr = defaultValueExpr
                };
            });

            return arguments.ToList();
        }

        public static List<SqlCodeExpr> ConsumeBeginBody(this IParser parser)
        {
            parser.Scanner.Consume(SqlToken.Begin);
            var bodyList = new List<SqlCodeExpr>();
            do
            {
                var body = parser.ParseExpIgnoreComment();
                if (body == null)
                {
                    break;
                }

                bodyList.Add(body as SqlCodeExpr);
            } while (!parser.Scanner.TryConsume(SqlToken.End, out _));

            parser.Scanner.Match(SqlToken.Semicolon);
            return bodyList;
        }

        public static List<SqlCodeExpr> ConsumeBeginBodyOrSingle(this IParser parser)
        {
            var bodyList = new List<SqlCodeExpr>();
            do
            {
                var body = parser.ParseExpIgnoreComment();
                if (body == null)
                {
                    break;
                }

                bodyList.Add(body);
            } while (true);

            return bodyList;
            //if (parser.Scanner.IsToken(SqlToken.Begin))
            //{
            //	return parser.ConsumeBeginBody();
            //}
            //var bodyList = new List<SqlCodeExpr>();
            //var body = parser.ParseExpIgnoreComment();
            //if (body == null)
            //{
            //	return bodyList;
            //}
            //bodyList.Add(body as SqlCodeExpr);
            //return bodyList;
        }

        public static IEnumerable<TExpression> ConsumeByDelimiter<TExpression>(this IParser parser,
            SqlToken delimiter,
            Func<TExpression> predicateExpr)
            where TExpression : SqlCodeExpr
        {
            parser.Scanner.IgnoreComments();
            return parser.ConsumeByDelimiter<SqlToken, TExpression>(delimiter, predicateExpr);
        }

        public static SqlCodeExpr ConsumeCreateProcedure(this IParser parser, TextSpan storeProcedureSpan)
        {
            var nameExpr = parser.ConsumeObjectId();
            var arguments = parser.ConsumeArgumentList();

            var withExecuteAsExpr = ParseWithExecuteAs(parser);

            parser.Scanner.Consume(SqlToken.As);
            var bodyList = parser.ConsumeBeginBodyOrSingle();

            return new CreateProcedureSqlCodeExpr
            {
                Name = nameExpr,
                Arguments = arguments,
                WithExecuteAs = withExecuteAsExpr,
                Body = bodyList
            };
        }

        private static WithExecuteAsSqlCodeExpr ParseWithExecuteAs(IParser parser)
        {
            if (!parser.MatchTokenList(SqlToken.With, SqlToken.Execute, SqlToken.As))
            {
                return null;
            }

            var userExpr =
                parser.ConsumeTokenStringAny(SqlToken.CALLER, SqlToken.SELF, SqlToken.OWNER, SqlToken.QuoteString);
            return new WithExecuteAsSqlCodeExpr
            {
                UserExpr = userExpr
            };
        }

        public static SqlCodeExpr ConsumeDataType(this IParser parser)
        {
            if (parser.Scanner.Match(SqlToken.TABLE))
            {
                return parser.ConsumeTableDataTypeList();
            }

            SqlCodeExpr dataType;
            if (parser.TryPrefixParseAny(int.MaxValue, out var userIdentifierDataType, SqlToken.Identifier,
                    SqlToken.SqlIdentifier))
            {
                dataType = userIdentifierDataType;
            }
            else
            {
                dataType = ParseDataType(parser);
            }

            var extraList = parser.ParseAll(
                ParseIdentity,
                ParseReadOnly,
                ParseDataTypeSize,
                ParseNotForReplication,
                SqlParserExtension.ParseClustered,
                ParsePrimaryKey,
                ParseConstraintWithOptions,
                ParseConstraint,
                ParseIsAllowNull,
                ParseDefault,
                ParseOnPrimary);

            return new DataTypeSqlCodeExpr
            {
                DataType = dataType,
                //IsIdentity = isIdentity,
                //IsReadOnly = isReadOnly,
                //SizeExpr = sizeExpr,
                ExtraList = extraList,
            };
        }

        public static TextSpan ConsumeIdentifierToken(this IParser parser)
        {
            if (!parser.TryConsumeIdentifierToken(out var identifier))
            {
                ThrowHelper.ThrowParseException(parser, "Expect Identifier");
            }

            return identifier;
        }

        public static ObjectIdSqlCodeExpr ParseMeetObjectId(this IParser parser)
        {
            var startIndex = parser.Scanner.GetOffset();

            var comments = parser.IgnoreComments();
            var tokens = new List<string>();
            do
            {
                var span = parser.ConsumeToken();
                if (parser.Scanner.IsSymbol(span) && span.GetTokenType() != SqlToken.Asterisk)
                {
                    break;
                }

                tokens.Add(parser.Scanner.GetSpanString(span));
                if (tokens.Count > 4)
                {
                    var prevTokens = string.Join(".", tokens);
                    var currTokenStr = parser.Scanner.PeekString();
                    throw new ParseException(
                        $"Expect RemoteServer.Database.dbo.name, but got too many Identifier at '{prevTokens}.{currTokenStr}'.");
                }
            } while (parser.Scanner.Match(SqlToken.Dot));

            if (tokens.Count == 0)
            {
                parser.Scanner.SetOffset(startIndex);
                return null;
            }

            var fixCount = 4 - tokens.Count;
            for (var i = 0; i < fixCount; i++)
            {
                tokens.Insert(0, string.Empty);
            }

            return new ObjectIdSqlCodeExpr
            {
                Comments = comments,
                RemoteServer = tokens[0],
                DatabaseName = tokens[1],
                SchemaName = tokens[2],
                ObjectName = tokens[3],
            };
        }

        public static SqlCodeExpr ConsumeObjectId(this IParser parser, bool nonSensitive = false)
        {
            if (!TryConsumeObjectId(parser, out var objectId, nonSensitive))
            {
                ThrowHelper.ThrowParseException(parser, "Expect ObjectId");
            }

            return objectId;
        }

        public static SqlCodeExpr ConsumeObjectIdOrVariable(this IParser parser, int ctxPrecedence = 0)
        {
            if (parser.TryConsumeObjectId(out var objectIdExpr))
            {
                return objectIdExpr;
            }

            return parser.PrefixParse(SqlToken.Variable, ctxPrecedence) as SqlCodeExpr;
        }

        public static SqlCodeExpr ConsumePrimary(this IParser parser)
        {
            if (parser.Scanner.TryConsumeAny(out var identifier, SqlToken.SqlIdentifier))
            {
                return parser.PrefixParse(identifier) as SqlCodeExpr;
            }

            parser.Scanner.Consume(SqlToken.PRIMARY);
            return new ObjectIdSqlCodeExpr
            {
                ObjectName = "PRIMARY"
            };
        }

        public static TableDataTypeSqlCodeExpr ConsumeTableDataTypeList(this IParser parser)
        {
            var columnDataTypeList = parser.ParseParenWithComma(() => parser.ParseAny(
                ParseConstraintExpr,
                ParseExtra,
                ParseColumnDefine), true).ToList();

            return new TableDataTypeSqlCodeExpr
            {
                Columns = columnDataTypeList
            };
        }

        public static SqlCodeExpr ConsumeTableName(this IParser parser, int ctxPrecedence = 0)
        {
            if (parser.TryConsumeTokenAny(out var nameSpan, SqlToken.TempTable, SqlToken.Variable))
            {
                var nameStr = parser.Scanner.GetSpanString(nameSpan);
                return new ObjectIdSqlCodeExpr
                {
                    ObjectName = nameStr
                };
            }

            if (!parser.TryConsumeObjectId(out var objectId))
            {
                var token = parser.PeekToken();
                if (parser.Scanner.IsSymbol(token))
                {
                    ThrowHelper.ThrowParseException(parser, "Expect TableName");
                }

                parser.ConsumeToken();
                return new ObjectIdSqlCodeExpr
                {
                    ObjectName = parser.Scanner.GetSpanString(token)
                };
            }

            return objectId;
        }

        public static TextSpan ConsumeToken(this IParser parser)
        {
            var token = PeekToken(parser);
            if (token.IsEmpty)
            {
                ThrowHelper.ThrowParseException(parser, "Unexpected None.");
            }

            if (token.IsComment())
            {
                ThrowHelper.ThrowParseException(parser, $"Except non-comment token, bot got {token.Type}.");
            }

            parser.Scanner.SetOffset(token.Offset + token.Length - 1);
            return token;
        }

        public static TextSpan ConsumeToken(this IParser parser, SqlToken tokenType)
        {
            var commentTokenTypes = new[]
            {
                SqlToken.MultiComment,
                SqlToken.SingleComment
            };
            var commentList = parser.Scanner.IgnoreComments();
            var token = parser.Scanner.Peek();
            if (commentTokenTypes.Contains(tokenType) && token.IsEmpty)
            {
                if (commentList.Count > 0)
                {
                    var allComment = commentList.First().Concat(commentList.Last());
                    return allComment;
                }

                ThrowHelper.ThrowParseException(parser, "Except Comment, but got None.");
            }

            if (token.Type != tokenType.ToString())
            {
                ThrowHelper.ThrowParseException(parser, $"Expect {tokenType}, but got {token.Type}");
            }

            parser.Scanner.Consume();
            token.Comments = commentList;
            return token;
        }

        public static TextSpan ConsumeTokenAny(this IParser parser, params SqlToken[] tokenTypeList)
        {
            var token = TextSpan.Empty;
            var isAny = tokenTypeList.Any(x => parser.TryConsumeToken(out token, x));
            if (!isAny)
            {
                var s = string.Join(",", tokenTypeList.Select(x => x.ToString()));
                ThrowHelper.ThrowParseException(parser, $"Expect one of {s}");
            }

            return token;
        }

        public static List<TextSpan> ConsumeTokenListAny(this IParser parser, params SqlToken[][] tokenTypeListArray)
        {
            var spanList = new List<TextSpan>();
            var isAny = tokenTypeListArray.Any(tokenTypeList =>
                parser.TryConsumeTokenList(out spanList, tokenTypeList));
            if (!isAny)
            {
                ThrowHelper.ThrowParseException(parser, "");
            }

            return spanList;
        }

        public static string ConsumeTokenString(this IParser parser)
        {
            var span = parser.ConsumeToken();
            return parser.Scanner.GetSpanString(span);
        }

        public static string ConsumeTokenStringAny(this IParser parser, params SqlToken[] tokenTypeList)
        {
            if (!parser.TryConsumeTokenAny(out var tokenSpan, tokenTypeList))
            {
                ThrowHelper.ThrowParseException(parser,
                    $"Expect TokenType {string.Join(",", tokenTypeList.Select(x => x.ToString()))}");
            }

            return parser.Scanner.GetSpanString(tokenSpan);
        }

        public static string ConsumeTokenStringListAny(this IParser parser, params SqlToken[][] tokenTypeListArray)
        {
            var spanList = parser.ConsumeTokenListAny(tokenTypeListArray);
            return string.Join(" ", spanList.Select(x => parser.Scanner.GetSpanString(x)));
        }

        public static SqlToken ConsumeTokenTypeAny(this IParser parser, params SqlToken[] tokenTypeList)
        {
            var span = ConsumeTokenAny(parser, tokenTypeList);
            return Enum.Parse<SqlToken>(span.Type);
        }

        public static TokenSqlCodeExpr ConsumeToTokenValueAny(this IParser parser, params SqlToken[] tokenTypeList)
        {
            var tokenType = parser.ConsumeTokenTypeAny(tokenTypeList);
            return new TokenSqlCodeExpr
            {
                Value = tokenType
            };
        }

        public static SqlCodeExpr ConsumeTrigger(this IParser parser)
        {
            parser.ConsumeToken(SqlToken.TRIGGER);

            SqlCodeExpr triggerName;
            if (parser.TryConsumeToken(out var allSpan, SqlToken.ALL))
            {
                triggerName = new AllSqlCodeExpr();
            }
            else
            {
                triggerName = parser.ConsumeObjectId();
            }

            parser.ConsumeToken(SqlToken.ON);

            SqlCodeExpr objectExpr;
            if (parser.MatchTokenList(SqlToken.ALL, SqlToken.SERVER))
            {
                objectExpr = new ObjectIdSqlCodeExpr
                {
                    ObjectName = "ALL SERVER"
                };
            }
            else if (parser.MatchToken(SqlToken.DATABASE))
            {
                objectExpr = new ObjectIdSqlCodeExpr
                {
                    ObjectName = "DATABASE"
                };
            }
            else
            {
                objectExpr = parser.ConsumeObjectId();
            }

            var withExecuteAsExpr = ParseWithExecuteAs(parser);
            var afterExpr = ParseAfter(parser);

            return new TriggerSqlCodeExpr
            {
                TriggerName = triggerName,
                ObjectExpr = objectExpr,
                WithExecuteAs = withExecuteAsExpr,
                AfterExpr = afterExpr
            };
        }

        private static SqlCodeExpr ParseAfter(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.AFTER))
            {
                return null;
            }

            var actionList = new List<SqlToken>();
            do
            {
                var actionName = parser.ConsumeToken().GetTokenType();
                actionList.Add(actionName);
            } while (parser.MatchToken(SqlToken.Comma));

            return new AfterActionSqlCodeExpr
            {
                ActionList = actionList
            };
        }

        public static ExprListSqlCodeExpr ConsumeValueList(this IParser parser)
        {
            parser.Scanner.Consume(SqlToken.LParen);
            var valueList = new List<SqlCodeExpr>();
            do
            {
                var valueExpr = parser.ParseExpIgnoreComment();
                valueList.Add(valueExpr);
            } while (parser.Scanner.Match(SqlToken.Comma));

            parser.Scanner.Consume(SqlToken.RParen);
            return new ExprListSqlCodeExpr
            {
                Items = valueList
            };
        }

        public static CreateTableSqlCodeExpr CreateTable(this IParser parser, TextSpan tableSpan)
        {
            var tableName = parser.ConsumeTableName();

            var tableType = parser.ConsumeTableDataTypeList();
            tableType.Name = tableName;

            var onPrimary = parser.ParseOnPrimary();

            var isSemicolon = parser.MatchToken(SqlToken.Semicolon);

            return new CreateTableSqlCodeExpr
            {
                TableExpr = tableType,
                OnPrimary = onPrimary,
                IsSemicolon = isSemicolon
            };
        }

        public static List<SqlCodeExpr> GetColumnsListExpr(this IParser parser)
        {
            var columnsList = new List<SqlCodeExpr>();
            if (parser.Scanner.Match(SqlToken.LParen))
            {
                do
                {
                    var column = parser.ParseExpIgnoreComment();
                    columnsList.Add(column);
                } while (parser.Scanner.Match(SqlToken.Comma));

                parser.Scanner.Consume(SqlToken.RParen);
            }

            return columnsList;
        }

        public static List<SqlCodeExpr> GetJoinSelectList(this IParser parser)
        {
            var joinSelectList = new List<SqlCodeExpr>();
            do
            {
                var crossApplyExpr = parser.ParseCrossApply();
                if (crossApplyExpr != null)
                {
                    joinSelectList.Add(crossApplyExpr);
                    continue;
                }

                if (parser.IsToken(SqlToken.Join))
                {
                    joinSelectList.Add(ParseJoinSelect(TextSpan.Empty, parser));
                    continue;
                }

                if (!parser.TryConsumeTokenAny(out var joinTypeSpan, SqlToken.Inner, SqlToken.Left, SqlToken.Right,
                        SqlToken.Full, SqlToken.Cross))
                {
                    break;
                }

                var joinSelect = ParseJoinSelect(joinTypeSpan, parser);
                joinSelectList.Add(joinSelect);
            } while (true);

            return joinSelectList;
        }

        public static Func<SqlCodeExpr> GetParseExpIgnoreCommentFunc(this IParser parser, int ctxPrecedence = 0)
        {
            var comments = new List<CommentSqlCodeExpr>();
            return () =>
            {
                SqlCodeExpr expr = null;
                while (true)
                {
                    var headToken = parser.PeekToken();
                    if (!headToken.IsEmpty && !parser.TryGetPrefixParselet(out _, headToken) &&
                        !parser.Scanner.IsSymbol(headToken))
                    {
                        var identifierToken = parser.ConsumeToken();
                        identifierToken.Type = SqlToken.Identifier.ToString();
                        expr = parser.PrefixParse(identifierToken) as SqlCodeExpr;
                    }
                    else
                    {
                        expr = parser.GetParseExp(ctxPrecedence) as SqlCodeExpr;
                    }

                    if (expr == null)
                    {
                        return null;
                    }

                    if (expr is CommentSqlCodeExpr commentExpr)
                    {
                        comments.Add(commentExpr);
                        continue;
                    }

                    expr.Comments = comments;
                    break;
                }

                return expr;
            };
        }

        public static List<TextSpan> IgnoreComments(this IScanner scanner)
        {
            var commentTypes = new[]
            {
                SqlToken.SingleComment.ToString(),
                SqlToken.MultiComment.ToString(),
            };
            var comments = new List<TextSpan>();
            do
            {
                var span = scanner.Peek();
                if (commentTypes.Contains(span.Type))
                {
                    scanner.Consume();
                    comments.Add(span);
                    continue;
                }

                break;
            } while (true);

            return comments;
        }

        public static List<CommentSqlCodeExpr> IgnoreComments(this IParser parser)
        {
            var commentsSpanList = parser.Scanner.IgnoreComments();
            return commentsSpanList.Select(x => new CommentSqlCodeExpr
            {
                Content = parser.Scanner.GetSpanString(x),
            }).ToList();
        }

        public static bool IsComment(this TextSpan span)
        {
            var commentTokenTypes = new[]
            {
                SqlToken.MultiComment.ToString(),
                SqlToken.SingleComment.ToString()
            };
            return commentTokenTypes.Contains(span.Type);
        }

        public static bool IsIdentifierToken(this IParser parser)
        {
            var startIndex = parser.Scanner.GetOffset();
            parser.TryConsumeToken(out var token0);
            parser.TryConsumeToken(out var token1);
            parser.Scanner.SetOffset(startIndex);

            if (token1.Type == SqlToken.LParen.ToString())
            {
                return false;
            }

            return !parser.Scanner.IsSymbol(token0) || token0.Type == SqlToken.Asterisk.ToString();
        }

        public static bool IsToken(this IParser parser, SqlToken tokenType)
        {
            var token = parser.PeekToken();
            return token.Type == tokenType.ToString();
        }

        public static bool Match(this IParser parser, SqlToken tokenType)
        {
            return parser.Scanner.Match(tokenType);
        }

        public static bool MatchToken(this IParser parser, SqlToken tokenType)
        {
            return TryConsumeToken(parser, out _, tokenType);
        }

        public static bool MatchTokenAny(this IParser parser, params SqlToken[] tokenTypeList)
        {
            var isAny = tokenTypeList.Any(x => parser.MatchToken(x));
            return isAny;
        }

        public static bool MatchTokenList(this IParser parser, params SqlToken[] tokenTypeList)
        {
            var startIndex = parser.Scanner.GetOffset();
            var isAll = tokenTypeList.All(x => parser.MatchToken(x));
            if (!isAll)
            {
                parser.Scanner.SetOffset(startIndex);
            }

            return isAll;
        }

        public static List<SqlCodeExpr> ParseAll(this IParser parser, params Func<IParser, SqlCodeExpr>[] parseFuncList)
        {
            var isAny = true;
            var exprList = new List<SqlCodeExpr>();
            var currentParseFuncList = new List<Func<IParser, SqlCodeExpr>>(parseFuncList);
            do
            {
                isAny = currentParseFuncList.Any(x =>
                {
                    var expr = x(parser);
                    var isMatch = expr != null;
                    if (isMatch)
                    {
                        exprList.Add(expr);
                        currentParseFuncList.Remove(x);
                    }

                    return isMatch;
                });
            } while (isAny);

            return exprList;
        }

        public static SqlCodeExpr ParseAny(this IParser parser, params Func<IParser, SqlCodeExpr>[] parseFuncList)
        {
            for (var i = 0; i < parseFuncList.Length; i++)
            {
                var parseFunc = parseFuncList[i];
                var expr = parseFunc(parser);
                if (expr != null)
                {
                    return expr;
                }
            }

            return null;
        }

        public static ClusteredSqlCodeExpr ParseClustered(this IParser parser)
        {
            if (!parser.TryConsumeTokenAny(out var headSpan, SqlToken.CLUSTERED, SqlToken.NONCLUSTERED))
            {
                return null;
            }

            var clusterType = parser.Scanner.GetSpanString(headSpan);
            var columnList = new List<OrderItemSqlCodeExpr>();

            if (parser.MatchToken(SqlToken.LParen))
            {
                columnList = parser.ParseOrderItemList();
                parser.ConsumeToken(SqlToken.RParen);
            }

            return new ClusteredSqlCodeExpr
            {
                ClusterType = clusterType,
                ColumnList = columnList,
            };
        }

        public static List<SqlCodeExpr> ParseColumnList(this IParser parser)
        {
            var columns = new List<SqlCodeExpr>();
            do
            {
                columns.Add(ConsumeColumnAs(parser));
            } while (parser.Match(SqlToken.Comma));

            return columns;
        }

        public static TextSpan ParseColumnNameToken(IParser parser)
        {
            var meetColumnNameList = new[]
            {
                SqlToken.SqlIdentifier, SqlToken.Identifier, SqlToken.QuoteString,
                SqlToken.TempTable,
                SqlToken.Source,
                SqlToken.Target,
                SqlToken.Asterisk,
                //
                SqlToken.Error,
                SqlToken.TYPE,
                SqlToken.Date,
                SqlToken.Rank
            };
            var span = parser.PeekToken();
            if (span.IsEmpty)
            {
                return TextSpan.Empty;
            }

            if (!meetColumnNameList.Contains(span.GetTokenType()))
            {
                return TextSpan.Empty;
            }

            return parser.ConsumeToken();
        }


        public static MarkConstraintSqlCodeExpr ParseConstraint(this IParser parser)
        {
            if (!parser.MatchToken(SqlToken.CONSTRAINT))
            {
                return null;
            }

            SqlCodeExpr name;
            if (parser.MatchToken(SqlToken.ALL))
            {
                name = new AllSqlCodeExpr();
            }
            else
            {
                name = parser.ConsumeObjectId();
            }

            return new MarkConstraintSqlCodeExpr
            {
                Name = name,
            };
        }

        public static SqlCodeExpr ParseConstraintWithOptions(this IParser parser)
        {
            if (!parser.MatchTokenList(SqlToken.With, SqlToken.LParen))
            {
                return null;
            }

            var optionList = new List<SqlCodeExpr>();
            do
            {
                var item = parser.ConsumeAny(ParseFillfactor, ParseToggle);
                optionList.Add(item);
            } while (parser.MatchToken(SqlToken.Comma));

            parser.ConsumeToken(SqlToken.RParen);

            return new ConstraintWithSqlCodeExpr
            {
                OptionList = optionList
            };
        }

        public static CrossApplySqlCodeExpr ParseCrossApply(this IParser parser)
        {
            if (!parser.MatchTokenList(SqlToken.Cross, SqlToken.APPLY))
            {
                return null;
            }

            // parser.ConsumeToken(SqlToken.LParen);
            // var expr = parser.ParseExpIgnoreComment();
            // parser.ConsumeToken(SqlToken.RParen);

            var expr = parser.ParseTableSource();

            parser.MatchToken(SqlToken.As);
            var aliasName = parser.ConsumeObjectId();

            var columnList = new List<SqlCodeExpr>();
            if (parser.MatchToken(SqlToken.LParen))
            {
                columnList = parser.ParseColumnList();
                parser.ConsumeToken(SqlToken.RParen);
            }

            return new CrossApplySqlCodeExpr
            {
                SubExpr = expr,
                AliasName = aliasName,
                ColumnList = columnList
            };
        }

        public static SqlCodeExpr ParseDefault(this IParser parser)
        {
            if (!parser.MatchToken(SqlToken.Default))
            {
                return null;
            }

            SqlCodeExpr valueExpr = null;

            // if (parser.TryConsumeTokenList(out var dateSpan,
            //         SqlToken.Number, SqlToken.Minus, SqlToken.Number, SqlToken.Minus, SqlToken.Number))
            // {
            //     var date = string.Join("", dateSpan.Select(x => parser.Scanner.GetSpanString(x)));
            //     parser.ConsumeToken(SqlToken.FOR);
            //     var forColumn = parser.ConsumeObjectId();
            //     valueExpr = new ValueForSqlCodeExpr
            //     {
            //         Value = new QuoteStringSqlCodeExpr
            //         {
            //             Value = date
            //         },
            //         Column = forColumn
            //     };
            // }
            if (parser.MatchToken(SqlToken.LParen))
            {
                valueExpr = parser.ParseExpIgnoreComment();
                parser.ConsumeToken(SqlToken.RParen);
            }
            else
            {
                valueExpr = parser.ParseExpIgnoreComment(int.MaxValue);
            }

            return new DefaultSqlCodeExpr
            {
                ValueExpr = valueExpr,
            };
        }

        public static SqlCodeExpr ParseExpIgnoreComment(this IParser parser, int ctxPrecedence = 0)
        {
            return parser.GetParseExpIgnoreCommentFunc(ctxPrecedence)();
        }

        public static List<SqlCodeExpr> ParseFromSourceList(this IParser parser)
        {
            var fromSourceList = new List<SqlCodeExpr>();
            do
            {
                fromSourceList.Add(ParseFromSource(parser));
            } while (parser.Scanner.Match(SqlToken.Comma));

            return fromSourceList;
        }

        public static TextSpan ParseFuncNameToken(this IParser parser)
        {
            var funcNameSpan = parser.PeekToken();
            var funcName = parser.Scanner.GetSpanString(funcNameSpan).ToUpper();
            if (!parser.Scanner.IsFuncName(funcName))
            {
                return TextSpan.Empty;
            }

            var symbolSpan = parser.PeekToken(1);
            if (!symbolSpan.IsTokenType(SqlToken.LParen))
            {
                return TextSpan.Empty;
            }

            parser.ConsumeToken();
            return funcNameSpan;
        }

        public static TextSpan ParseFuncNameIdentifierToken(this IParser parser)
        {
            var funcNameSpan = parser.PeekToken();
            var funcName = parser.Scanner.GetSpanString(funcNameSpan).ToUpper();
            if (!parser.Scanner.IsFuncName(funcName))
            {
                return TextSpan.Empty;
            }

            var symbolSpan = parser.PeekToken(1);
            if (symbolSpan.IsTokenType(SqlToken.LParen))
            {
                return TextSpan.Empty;
            }

            var span = parser.ConsumeToken();
            span.Type = SqlToken.Identifier.ToString();
            return span;
        }

        public static SqlCodeExpr ParseInfix(this IParser parser, SqlCodeExpr leftExpr, TextSpan tokenSpan)
        {
            if (!parser.TryGetInfixParselet(out var infixParselet, tokenSpan))
            {
                ThrowHelper.ThrowParseException(parser, "Not found");
            }

            return infixParselet.Parse(leftExpr, tokenSpan, parser) as SqlCodeExpr;
        }

        public static SqlCodeExpr ParseInto_TableExpr(this IParser parser)
        {
            if (!parser.Scanner.Match(SqlToken.Into))
            {
                return null;
            }

            var intoTable = parser.ConsumeObjectIdOrVariable();

            var columnsList = new List<SqlCodeExpr>();
            if (parser.MatchToken(SqlToken.LParen))
            {
                do
                {
                    var columnName = parser.ConsumeAny(SqlToken.Identifier, SqlToken.SqlIdentifier) as SqlCodeExpr;
                    columnsList.Add(columnName);
                } while (parser.MatchToken(SqlToken.Comma));

                parser.ConsumeToken(SqlToken.RParen);
            }

            return new OutputIntoSqlCodeExpr
            {
                IntoTable = intoTable,
                ColumnsList = columnsList
            };
        }

        public static SqlCodeExpr ParseLRParenExpr(this IParser parser, SqlCodeExpr leftExpr, bool reqColumns = false)
        {
            var parameters = ParseLRParen(parser, reqColumns);
            if (parameters == null)
            {
                return leftExpr;
            }

            var callExpr = new FuncSqlCodeExpr
            {
                Name = leftExpr,
                Parameters = parameters
            };

            return parser.PrefixParse(callExpr) as SqlCodeExpr;
        }

        public static OnSqlCodeExpr ParseOnPrimary(this IParser parser)
        {
            if (!parser.MatchToken(SqlToken.ON))
            {
                return null;
            }

            var name = parser.ConsumeObjectId();
            return new OnSqlCodeExpr
            {
                Name = name
            };
        }

        public static OptionSqlCodeExpr ParseOptionExpr(this IParser parser)
        {
            if (!parser.Scanner.Match(SqlToken.Option))
            {
                return null;
            }

            parser.Scanner.Consume(SqlToken.LParen);

            parser.Scanner.Consume(SqlToken.MAXDOP);
            var numberOfCpu = int.Parse(parser.Scanner.ConsumeString(SqlToken.Number));
            parser.Scanner.Consume(SqlToken.RParen);

            var maxdop = new MaxdopSqlCodeExpr
            {
                NumberOfCpu = numberOfCpu,
            };

            return new OptionSqlCodeExpr
            {
                Maxdop = maxdop
            };
        }

        public static List<OrderItemSqlCodeExpr> ParseOrderItemList(this IParser parser)
        {
            var orderItemList = new List<OrderItemSqlCodeExpr>();
            do
            {
                var name = parser.ParseExpIgnoreComment();

                var ascOrDesc = "ASC";
                parser.Scanner.TryConsumeAny(out var ascOrDescSpan, SqlToken.Asc, SqlToken.Desc);
                if (!ascOrDescSpan.IsEmpty)
                {
                    ascOrDesc = parser.Scanner.GetSpanString(ascOrDescSpan);
                }

                orderItemList.Add(new OrderItemSqlCodeExpr
                {
                    Name = name,
                    AscOrDesc = ascOrDesc,
                });
            } while (parser.Scanner.Match(SqlToken.Comma));

            return orderItemList;
        }

        public static List<SqlCodeExpr> ParseOutputListExpr(this IParser parser)
        {
            var outputList = new List<SqlCodeExpr>();
            if (parser.Scanner.Match(SqlToken.Output))
            {
                do
                {
                    if (parser.Scanner.TryConsumeStringAny(out var actionName, SqlToken.Deleted, SqlToken.Inserted))
                    {
                        parser.Scanner.Consume(SqlToken.Dot);
                    }

                    var columnName = parser.ParseExpIgnoreComment();

                    parser.TryConsumeAliasName(out var aliasName);

                    outputList.Add(new OutputSqlCodeExpr
                    {
                        OutputActionName = actionName,
                        ColumnName = columnName,
                        AliasName = aliasName
                    });
                } while (parser.Scanner.Match(SqlToken.Comma));
            }

            return outputList;
        }

        public static IEnumerable<T> ParseParenWithComma<T>(this IParser parser, Func<T> parseItem,
            bool canIgnoreCommaAtLast = false)
        {
            if (!parser.MatchToken(SqlToken.LParen))
            {
                yield break;
            }

            do
            {
                if (parser.IsToken(SqlToken.RParen))
                {
                    break;
                }

                var expr = parseItem();
                yield return expr;
            } while (parser.MatchToken(SqlToken.Comma));

            if (canIgnoreCommaAtLast && !parser.IsToken(SqlToken.RParen))
            {
                var lastExpr = parseItem();
                if (lastExpr != null)
                {
                    yield return lastExpr;
                }
            }

            parser.ConsumeToken(SqlToken.RParen);
        }

        public static SqlCodeExpr ParsePrefix(this IParser parser, SqlToken tokenType)
        {
            if (!parser.TryConsumeToken(out var tokenSpan, tokenType))
            {
                return null;
            }

            return parser.PrefixParse(tokenSpan) as SqlCodeExpr;
        }

        public static PrimaryKeySqlCodeExpr ParsePrimaryKey(IParser parser)
        {
            if (!parser.Scanner.IsTokenList(SqlToken.PRIMARY, SqlToken.KEY))
            {
                return null;
            }

            parser.Scanner.Consume(SqlToken.PRIMARY);
            parser.Scanner.Consume(SqlToken.KEY);

            var columnList = new List<SqlCodeExpr>();
            if (parser.MatchToken(SqlToken.LParen))
            {
                do
                {
                    var column = parser.ConsumeObjectId();
                    columnList.Add(column);
                } while (parser.MatchToken(SqlToken.Comma));

                parser.ConsumeToken(SqlToken.RParen);
            }

            return new PrimaryKeySqlCodeExpr()
            {
                ColumnList = columnList
            };
        }

        public static TopSqlCodeExpr ParseTopCountExpr(this IParser parser)
        {
            if (!parser.Scanner.Match(SqlToken.Top))
            {
                return null;
            }

            SqlCodeExpr topNumberExpr;
            if (parser.MatchToken(SqlToken.LParen))
            {
                topNumberExpr = parser.ParseExpIgnoreComment();
                topNumberExpr = new GroupSqlCodeExpr
                {
                    InnerExpr = topNumberExpr,
                };
                parser.ConsumeToken(SqlToken.RParen);
            }
            else
            {
                topNumberExpr = parser.ParseExpIgnoreComment(int.MaxValue);
            }

            var isPercent = parser.MatchToken(SqlToken.PERCENT);

            return new TopSqlCodeExpr
            {
                NumberExpr = topNumberExpr,
                IsPercent = isPercent
            };
        }

        public static TypeObjectIdSqlCodeExpr ParseTypeObjectId(this IParser parser)
        {
            if (!parser.MatchTokenList(SqlToken.TYPE, SqlToken.ColonColon))
            {
                return null;
            }

            var name = parser.ConsumeObjectId();
            return new TypeObjectIdSqlCodeExpr
            {
                ObjectId = name,
            };
        }

        public static List<string> ParseWithOption(this IParser parser)
        {
            var userWithOptions = new List<string>();
            if (parser.Scanner.Match(SqlToken.With))
            {
                userWithOptions = parser.ParseWithOptionItemList();
            }

            return userWithOptions;
        }

        public static List<string> ParseWithOptionItemList(this IParser parser)
        {
            var userWithOptions = new List<string>();
            if (!parser.MatchToken(SqlToken.LParen))
            {
                return userWithOptions;
            }

            var withOptions = new[]
            {
                SqlToken.NOLOCK,
                SqlToken.ROWLOCK,
                SqlToken.UPDLOCK,
                SqlToken.HOLDLOCK,
                SqlToken.FORCESEEK,
                SqlToken.NOWAIT
            };

            do
            {
                if (parser.Scanner.Match(SqlToken.Index))
                {
                    parser.Scanner.Consume(SqlToken.LParen);
                    var indexName = parser.ConsumeObjectId();
                    parser.Scanner.Consume(SqlToken.RParen);
                    userWithOptions.Add($"INDEX({indexName})");
                    continue;
                }

                var option = parser.Scanner.ConsumeStringAny(withOptions);
                userWithOptions.Add(option);
            } while (parser.Scanner.Match(SqlToken.Comma));

            parser.Scanner.Consume(SqlToken.RParen);

            return userWithOptions;
        }

        public static TextSpan PeekToken(this IParser parser)
        {
            var startIndex = parser.Scanner.GetOffset();
            var commentList = parser.Scanner.IgnoreComments();
            var token = parser.Scanner.Consume();
            parser.Scanner.SetOffset(startIndex);
            if (token.IsEmpty)
            {
                if (commentList.Count > 0)
                {
                    var allComment = commentList.First().Concat(commentList.Last());
                    allComment.Type = SqlToken.MultiComment.ToString();
                    return allComment;
                }

                return TextSpan.Empty;
            }

            token.Comments = commentList;
            return token;
        }

        public static TextSpan PeekToken(this IParser parser, int n)
        {
            var startIndex = parser.Scanner.GetOffset();
            var token = TextSpan.Empty;
            for (var i = 0; i <= n; i++)
            {
                if (!parser.TryConsumeToken(out token))
                {
                    break;
                }
            }

            parser.Scanner.SetOffset(startIndex);
            return token;
        }

        public static SqlCodeExpr PrefixParseAny(this IParser parser, int ctxPrecedence,
            params SqlToken[] prefixTokenTypeList)
        {
            if (!parser.TryPrefixParseAny(ctxPrecedence, out SqlCodeExpr expr, prefixTokenTypeList))
            {
                ThrowHelper.ThrowParseException(parser, "");
            }

            return expr;
        }

        public static bool TryConsume(this IParser parser, out SqlCodeExpr expr, Func<TextSpan, SqlCodeExpr> toExpr,
            SqlToken tokenType)
        {
            return parser.TryConsumeAny(out expr, toExpr, tokenType);
        }

        public static bool TryConsume(this IParser parser, SqlToken tokenType, int ctxPrecedence, out SqlCodeExpr expr)
        {
            return parser.TryConsumeAny(ctxPrecedence, out expr, tokenType);
        }

        public static bool TryConsume(this IParser parser, SqlToken tokenType, out SqlCodeExpr expr)
        {
            return parser.TryConsumeAny(0, out expr, tokenType);
        }

        public static bool TryConsumeAliasName(this IParser parser, out SqlCodeExpr aliasNameExpr)
        {
            if (parser.TryConsumeObjectId(out aliasNameExpr))
            {
                return true;
            }

            var startIndex = parser.Scanner.GetOffset();
            if (parser.Scanner.Match(SqlToken.As))
            {
                var success = parser.TryConsumeObjectId(out aliasNameExpr);
                if (!success)
                {
                    parser.Scanner.SetOffset(startIndex);
                }

                return success;
            }

            aliasNameExpr = null;
            return false;
        }

        public static bool TryConsumeAny(this IParser parser, out SqlCodeExpr expr, Func<TextSpan, SqlCodeExpr> toExpr,
            params SqlToken[] tokenTypeList)
        {
            var comments = parser.Scanner.IgnoreComments();
            for (var i = 0; i < tokenTypeList.Length; i++)
            {
                var tokenType = tokenTypeList[i];
                if (parser.Scanner.TryConsume(tokenType, out var token))
                {
                    token.Comments = comments;
                    expr = toExpr(token);
                    return true;
                }
            }

            expr = null;
            return false;
        }

        public static bool TryConsumeAny(this IParser parser, int ctxPrecedence, out SqlCodeExpr expr,
            params SqlToken[] tokenTypeList)
        {
            return parser.TryConsumeAny(out expr, (span) => parser.PrefixParse(span, ctxPrecedence) as SqlCodeExpr,
                tokenTypeList);
        }

        public static bool TryConsumeAny(this IParser parser, out SqlCodeExpr expr, params SqlToken[] tokenTypeList)
        {
            return parser.TryConsumeAny(out expr, (span) => parser.PrefixParse(span, int.MaxValue) as SqlCodeExpr,
                tokenTypeList);
        }

        public static bool TryConsumeIdentifierToken(this IParser parser, out TextSpan identifier)
        {
            return parser.TryConsumeTokenAny(out identifier,
                ParseColumnNameToken,
                ParseFuncNameIdentifierToken);
        }

        public static bool TryConsumeObjectId(this IParser parser, out SqlCodeExpr expr, bool nonSensitive = false)
        {
            var comments = parser.IgnoreComments();

            var typeObjectId = parser.ParseTypeObjectId();
            if (typeObjectId != null)
            {
                expr = typeObjectId;
                return true;
            }

            var identTokens = new List<string>();
            do
            {
                if (identTokens.Count == 1 && parser.Scanner.IsToken(SqlToken.Dot))
                {
                    identTokens.Add("dbo");
                    continue;
                }

                if (identTokens.Count >= 4)
                {
                    var prevTokens = string.Join(".", identTokens);
                    var currTokenStr = parser.Scanner.PeekString();
                    throw new ParseException(
                        $"Expect RemoteServer.Database.dbo.name, but got too many Identifier at '{prevTokens}.{currTokenStr}'.");
                }

                var identifier = TextSpan.Empty;
                if (nonSensitive)
                {
                    //if (parser.Scanner.IsSymbol() && !parser.IsToken(SqlToken.Asterisk))
                    if (parser.Scanner.IsSymbol())
                    {
                        break;
                    }

                    identifier = parser.Scanner.Consume();
                }
                else if (!parser.TryConsumeIdentifierToken(out identifier))
                {
                    break;
                }

                identTokens.Add(parser.Scanner.GetSpanString(identifier));
            } while (parser.Scanner.Match(SqlToken.Dot));

            if (identTokens.Count == 0)
            {
                expr = null;
                return false;
            }

            var fixCount = 4 - identTokens.Count;
            for (var i = 0; i < fixCount; i++)
            {
                identTokens.Insert(0, string.Empty);
            }

            var identExpr = new ObjectIdSqlCodeExpr
            {
                Comments = comments,
                RemoteServer = identTokens[0],
                DatabaseName = identTokens[1],
                SchemaName = identTokens[2],
                ObjectName = identTokens[3],
            };

            expr = identExpr;
            return true;
        }

        public static bool TryConsumeToken(this IParser parser, out TextSpan token, SqlToken tokenType)
        {
            var span = parser.PeekToken();
            if (span.Type != tokenType.ToString())
            {
                token = TextSpan.Empty;
                return false;
            }

            parser.ConsumeToken();
            token = span;
            return true;
        }

        public static bool TryConsumeToken(this IParser parser, out TextSpan token)
        {
            token = PeekToken(parser);
            if (token.IsEmpty)
            {
                return false;
            }

            if (token.IsComment())
            {
                return false;
            }

            parser.Scanner.SetOffset(token.Offset + token.Length - 1);
            return true;
        }

        public static bool TryConsumeTokenAny(this IParser parser, out TextSpan span,
            params Func<IParser, TextSpan>[] parseTokenFuncList)
        {
            var tmpSpan = TextSpan.Empty;
            var isAny = parseTokenFuncList.Any(x =>
            {
                tmpSpan = x(parser);
                return !tmpSpan.IsEmpty;
            });
            span = tmpSpan;
            return isAny;
        }

        public static bool TryConsumeTokenAny(this IParser parser, out TextSpan token, params SqlToken[] tokenTypeList)
        {
            TextSpan tmpToken = TextSpan.Empty;
            var isSuccess = tokenTypeList.Any(tokenType => parser.TryConsumeToken(out tmpToken, tokenType));
            token = tmpToken;
            return isSuccess;
        }

        public static bool TryConsumeTokenList(this IParser parser, out List<TextSpan> spanList,
            params SqlToken[] tokenTypeList)
        {
            var startIndex = parser.Scanner.GetOffset();
            var list = new List<TextSpan>();
            var isAll = tokenTypeList.All(x =>
            {
                var isSuccess = parser.TryConsumeToken(out var span, x);
                if (isSuccess)
                {
                    list.Add(span);
                }

                return isSuccess;
            });
            if (!isAll)
            {
                list = new List<TextSpan>();
                parser.Scanner.SetOffset(startIndex);
            }

            spanList = list;
            return isAll;
        }

        public static bool TryConsumeVariable(this IScanner scanner, out VariableSqlCodeExpr sqlExpr)
        {
            if (!scanner.TryConsume(SqlToken.Variable, out var returnVariableSpan))
            {
                sqlExpr = null;
                return false;
            }

            sqlExpr = new VariableSqlCodeExpr
            {
                Name = scanner.GetSpanString(returnVariableSpan)
            };
            return true;
        }

        public static bool TryPrefixParseAny(this IParser parser, int ctxPrecedence, out SqlCodeExpr expr,
            params SqlToken[] prefixTokenTypeList)
        {
            var prefixTokenTypeStrList = prefixTokenTypeList.Select(x => x.ToString()).ToArray();
            var startIndex = parser.Scanner.GetOffset();
            var prefixToken = parser.Scanner.Consume();
            if (!prefixTokenTypeStrList.Contains(prefixToken.Type))
            {
                parser.Scanner.SetOffset(startIndex);
                expr = null;
                return false;
            }

            expr = parser.PrefixParse(prefixToken, ctxPrecedence) as SqlCodeExpr;
            return true;
        }

        public static void WriteToStream<T>(this IEnumerable<T> exprList, IndentStream stream,
            Action<IndentStream> writeDelimiter = null)
        {
            if (writeDelimiter == null)
            {
                writeDelimiter = (stream1) => stream1.WriteLine();
            }

            foreach (var expr in exprList.Select((val, idx) => new {val, idx}))
            {
                if (expr.idx != 0)
                {
                    writeDelimiter(stream);
                }

                if (expr.val is SqlCodeExpr sqlExpr)
                {
                    //expr.val.WriteToStream(stream);
                    sqlExpr.WriteToStream(stream);
                }
                else
                {
                    stream.Write($"{expr.val}");
                }
            }
        }

        public static IEnumerable<string> ToUpper<T>(this IEnumerable<T> exprList)
        {
            return exprList.Select(x => $"{x}".ToUpper());
        }

        public static void WriteToStreamWithComma<T>(this IEnumerable<T> exprList, IndentStream stream)
        {
            exprList.WriteToStream(stream, s => stream.Write(", "));
        }

        public static void WriteToStreamWithCommaLine(this IEnumerable<SqlCodeExpr> exprList, IndentStream stream)
        {
            foreach (var expr in exprList.Select((val, idx) => new {val, idx}))
            {
                if (expr.idx != 0)
                {
                    stream.WriteLine(",");
                }

                expr.val.WriteToStream(stream);
            }
        }

        private static SqlCodeExpr Consume(this IParser parser, TryConsumeDelegate predicate)
        {
            if (!predicate(parser, out var expr))
            {
                ThrowHelper.ThrowParseException(parser, string.Empty);
            }

            return expr;
        }

        private static SqlCodeExpr Parse(IParser parser, SqlToken tokenType)
        {
            if (!parser.MatchToken(tokenType))
            {
                return null;
            }

            return new TokenSqlCodeExpr
            {
                Value = tokenType,
            };
        }

        public static SqlCodeExpr ParseMeetColumnName(this IParser parser)
        {
            var head = parser.PeekToken();
            if (parser.TryGetPrefixParselet(out var prefix, head))
            {
                parser.ConsumeToken();
                return prefix.Parse(head, parser) as SqlCodeExpr;
            }

            var funcNameSpan = parser.ParseFuncNameToken();
            if (!funcNameSpan.IsEmpty && parser.IsToken(SqlToken.LParen))
            {
                return parser.PrefixParse(funcNameSpan) as SqlCodeExpr;
            }

            if (parser.TryConsumeToken(out var varNameSpan, SqlToken.Variable))
            {
                return parser.PrefixParse(varNameSpan, int.MaxValue) as SqlCodeExpr;
            }

            if (parser.TryConsumeToken(out var asteriskSpan, SqlToken.Asterisk))
            {
                return parser.PrefixParse(asteriskSpan, int.MaxValue) as SqlCodeExpr;
            }

            SqlCodeExpr expr = parser.ParseMeetObjectId();
            if (expr != null)
            {
                return parser.ParseLRParenExpr(expr);
            }

            return parser.ParseExpIgnoreComment(int.MaxValue);
        }

        public static AssignSqlCodeExpr ParseAssign(this IParser parser, SqlCodeExpr leftExpr)
        {
            if (!parser.MatchToken(SqlToken.Equal))
            {
                return null;
            }

            var expr = parser.ParseExpIgnoreComment();

            return new AssignSqlCodeExpr
            {
                Left = leftExpr,
                Right = expr
            };
        }

        private static SqlCodeExpr ConsumeColumnAs(IParser parser)
        {
            var name = parser.ParseExpIgnoreComment();
            name = parser.ParseLRParenExpr(name);

            // SqlCodeExpr name = parser.ParseMeetColumnName();
            // name = parser.ParseLRParenExpr(name);

            var assignExpr = parser.ParseAssign(name);
            if (assignExpr != null)
            {
                name = assignExpr;
            }

            SqlCodeExpr overExpr = null;
            if (parser.TryConsumeToken(out var overSpan, SqlToken.Over))
            {
                overExpr = parser.PrefixParse(overSpan) as SqlCodeExpr;
            }

            var hasAs = parser.Scanner.Match(SqlToken.As);

            SqlCodeExpr aliasName = null;
            if (hasAs)
            {
                var aliasNameToken = parser.ConsumeToken();
                aliasName = new ObjectIdSqlCodeExpr
                {
                    ObjectName = parser.Scanner.GetSpanString(aliasNameToken)
                };
            }
            else
            {
                parser.TryConsumeObjectId(out aliasName);
            }

            return new ColumnSqlCodeExpr
            {
                Name = name,
                OverExpr = overExpr,
                AliasName = aliasName
            };
        }

        private static ColumnDefineSqlCodeExpr ParseColumnDefine(IParser parser)
        {
            if (!parser.TryConsumeIdentifierToken(out var nameSpan))
            {
                return null;
            }

            var name = parser.Scanner.GetSpanString(nameSpan);

            var dataType = parser.ConsumeDataType();
            var columnDefineSqlCodeExpr = new ColumnDefineSqlCodeExpr
            {
                Name = name,
                DataType = dataType,
            };
            return columnDefineSqlCodeExpr;
        }

        private static SqlCodeExpr ParseConstraintExpr(IParser parser)
        {
            if (!parser.TryConsumeToken(out var constraintSpan, SqlToken.CONSTRAINT))
            {
                return null;
            }

            return parser.PrefixParse(constraintSpan) as SqlCodeExpr;
        }

        private static ObjectIdSqlCodeExpr ParseDataType(IParser parser)
        {
            var allTypes = TSqlParser.DataTypes.Concat(new[] {SqlToken.Identifier}).ToArray();
            var dataTypeToken = parser.Scanner.ConsumeAny(allTypes);
            var dataTypeStr = parser.Scanner.GetSpanString(dataTypeToken);
            if (TSqlParser.DataTypes.Select(x => x.ToString()).Contains(dataTypeToken.Type))
            {
                dataTypeStr = dataTypeStr.ToUpper();
            }

            return new ObjectIdSqlCodeExpr
            {
                ObjectName = dataTypeStr,
            };
        }

        private static DataTypeSizeSqlCodeExpr ParseDataTypeSize(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.LParen))
            {
                return null;
            }

            SqlCodeExpr size;
            if (parser.MatchToken(SqlToken.MAX))
            {
                size = new MaxSqlCodeExpr();
            }
            else
            {
                size = new NumberSqlCodeExpr
                {
                    Value = parser.ConsumeTokenStringAny(SqlToken.Number)
                };
            }

            int? scale = null;
            if (parser.Scanner.Match(SqlToken.Comma))
            {
                var scaleToken = parser.Scanner.Consume(SqlToken.Number);
                scale = int.Parse(parser.Scanner.GetSpanString(scaleToken));
            }

            parser.Scanner.Consume(SqlToken.RParen);

            return new DataTypeSizeSqlCodeExpr
            {
                Size = size,
                Scale = scale,
            };
        }

        private static SqlCodeExpr ParseExtra(IParser parser)
        {
            var list = parser.ParseAll(
                SqlParserExtension.ParseClustered,
                ParsePrimaryKey,
                ParseConstraintWithOptions,
                ParseOnPrimary);
            if (list.Count == 0)
            {
                return null;
            }

            return new ExprListSqlCodeExpr
            {
                IsComma = false,
                Items = list,
            };
        }

        private static FillfactorSqlCodeExpr ParseFillfactor(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.FILLFACTOR))
            {
                return null;
            }

            parser.ConsumeToken(SqlToken.Equal);
            var fillfactorValue = parser.Consume(SqlToken.Number);
            var fillfactorSqlCodeExpr = new FillfactorSqlCodeExpr
            {
                Value = fillfactorValue
            };
            return fillfactorSqlCodeExpr;
        }

        private static SqlCodeExpr ParseFromSource(IParser parser)
        {
            // var isLParen = parser.MatchToken(SqlToken.LParen);
            // SqlCodeExpr sourceExpr;
            // if (isLParen)
            // {
            //     sourceExpr = parser.ParseExpIgnoreComment();
            //     parser.ConsumeToken(SqlToken.RParen);
            //     sourceExpr = new GroupSqlCodeExpr
            //     {
            //         InnerExpr = sourceExpr
            //     };
            // }
            // else
            // {
            //     sourceExpr = parser.ParseTableSource();
            //     if (sourceExpr == null)
            //     {
            //         ThrowHelper.ThrowParseException(parser, "TableName");
            //     }
            // }

            var sourceExpr = parser.ParseTableSource();
            if (sourceExpr == null)
            {
                ThrowHelper.ThrowParseException(parser, "TableName");
            }

            return parser.ConsumeAfterTableSource(sourceExpr);
        }

        private static SqlCodeExpr ConsumeAfterTableSource(this IParser parser, SqlCodeExpr sourceExpr)
        {
            parser.TryConsumeAliasName(out var aliasNameExpr);
            var userWithOptions = parser.ParseWithOption();
            var joinList = parser.GetJoinSelectList();
            return new FromSourceSqlCodeExpr
            {
                Left = sourceExpr,
                AliasName = aliasNameExpr,
                Options = userWithOptions,
                JoinList = joinList
            };
        }

        private static SqlCodeExpr ParseIdentity(IParser parser)
        {
            return Parse(parser, SqlToken.IDENTITY);
        }

        private static SqlCodeExpr ParseIsAllowNull(IParser parser)
        {
            if (parser.MatchTokenList(SqlToken.Not, SqlToken.Null))
            {
                return new NotNullSqlCodeExpr();
            }

            if (parser.Scanner.Match(SqlToken.Null))
            {
                return new NullSqlCodeExpr();
            }

            return null;
        }

        private static SqlCodeExpr ParseJoinSelect(TextSpan joinTypeSpan, IParser parser)
        {
            var parselet = new JoinParselet();
            return parselet.Parse(joinTypeSpan, parser) as SqlCodeExpr;
        }

        private static List<SqlCodeExpr> ParseLRParen(IParser parser, bool reqColumns = false)
        {
            var parameterList = new List<SqlCodeExpr>();
            if (!parser.Scanner.Match(SqlToken.LParen))
            {
                return null;
            }

            var startIndex = parser.Scanner.GetOffset();
            do
            {
                if (parser.IsToken(SqlToken.RParen))
                {
                    break;
                }

                var parameter = parser.ParseExpIgnoreComment();
                parameterList.Add(parameter);
            } while (parser.MatchToken(SqlToken.Comma));

            parser.ConsumeToken(SqlToken.RParen);

            if (reqColumns)
            {
                var isAllObjectId = parameterList.All(x => x is ObjectIdSqlCodeExpr);
                if (!isAllObjectId)
                {
                    parser.Scanner.SetOffset(startIndex);
                    return new List<SqlCodeExpr>();
                }
            }

            return parameterList;
        }

        private static NonClusteredSqlCodeExpr ParseNonClustered(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.NONCLUSTERED))
            {
                return null;
            }

            return new NonClusteredSqlCodeExpr();
        }

        private static NotForReplicationSqlCodeExpr ParseNotForReplication(IParser parser)
        {
            if (!parser.MatchTokenList(SqlToken.Not, SqlToken.FOR, SqlToken.REPLICATION))
            {
                return null;
            }

            return new NotForReplicationSqlCodeExpr();
        }

        private static ExprListSqlCodeExpr ParseParametersList(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.LParen))
            {
                return null;
            }

            var parametersList = new List<SqlCodeExpr>();
            do
            {
                parametersList.Add(parser.ParseExpIgnoreComment());
            } while (parser.MatchToken(SqlToken.Comma));

            parser.ConsumeToken(SqlToken.RParen);
            return new ExprListSqlCodeExpr
            {
                Items = parametersList
            };
        }

        private static SqlCodeExpr ParseReadOnly(IParser parser)
        {
            return Parse(parser, SqlToken.ReadOnly);
        }

        private static int? ParseSize(IParser parser)
        {
            int? size = null;
            if (parser.Scanner.Match(SqlToken.MAX))
            {
                size = int.MaxValue;
            }
            else
            {
                var sizeToken = parser.Scanner.Consume(SqlToken.Number);
                var sizeStr = parser.Scanner.GetSpanString(sizeToken);
                size = int.Parse(sizeStr);
            }

            return size;
        }

        public static SqlCodeExpr ParseTableSource(this IParser parser)
        {
            ObjectIdSqlCodeExpr ParseFuncNameIdentifier(IParser parser1)
            {
                var funcNameSpan = parser1.ParseFuncNameIdentifierToken();
                if (funcNameSpan.IsEmpty)
                {
                    return null;
                }

                var funcName = parser1.Scanner.GetSpanString(funcNameSpan);
                return new ObjectIdSqlCodeExpr
                {
                    ObjectName = funcName
                };
            }

            FuncSqlCodeExpr ParseIdentifierFunc(IParser parser1)
            {
                var startIndex = parser1.Scanner.GetOffset();
                if (!parser1.TryConsumeObjectId(out var objectId))
                {
                    return null;
                }

                if (!parser1.IsToken(SqlToken.LParen))
                {
                    parser1.Scanner.SetOffset(startIndex);
                    return null;
                }

                var parametersList = ParseParametersList(parser1);

                return new FuncSqlCodeExpr
                {
                    Name = objectId,
                    Parameters = parametersList.Items
                };
            }

            SqlCodeExpr ParseVariable(IParser parser1)
            {
                if (!parser1.TryConsumeToken(out var variableSpan, SqlToken.Variable))
                {
                    return null;
                }

                return parser1.PrefixParse(variableSpan, int.MaxValue) as SqlCodeExpr;
            }

            ObjectIdSqlCodeExpr ParseTempTable(IParser parser1)
            {
                if (!parser1.TryConsumeToken(out var tempTableSpan, SqlToken.TempTable))
                {
                    return null;
                }

                return new ObjectIdSqlCodeExpr
                {
                    ObjectName = parser1.Scanner.GetSpanString(tempTableSpan)
                };
            }

            SqlCodeExpr ParseIdentifier(IParser parser1)
            {
                if (!parser1.TryConsumeObjectId(out var identifier))
                {
                    return null;
                }

                return parser.ParseLRParenExpr(identifier);
            }

            SqlCodeExpr ParseNonIdentifier(IParser parser1)
            {
                var span = parser1.PeekToken();
                if (parser1.Scanner.IsSymbol(span))
                {
                    return null;
                }

                parser1.ConsumeToken();
                return new ObjectIdSqlCodeExpr
                {
                    ObjectName = parser1.Scanner.GetSpanString(span)
                };
            }

            SqlCodeExpr ParseRent(IParser parser1)
            {
                if (!parser1.MatchToken(SqlToken.LParen))
                {
                    return null;
                }

                var innerExpr = parser1.ParseExpIgnoreComment();
                parser1.ConsumeToken(SqlToken.RParen);
                return new GroupSqlCodeExpr
                {
                    InnerExpr = innerExpr
                };
            }
            
            SqlCodeExpr ParseRentTableWithOption(IParser parser1)
            {
                var startIndex = parser.Scanner.GetOffset();
                if (!parser1.MatchToken(SqlToken.LParen))
                {
                    return null;
                }
                if (!parser1.TryConsumeObjectId(out var tableName))
                {
                    parser1.Scanner.SetOffset(startIndex);
                    return null;
                }
                if (!parser1.IsToken(SqlToken.With))
                {
                    parser1.Scanner.SetOffset(startIndex);
                    return null;
                }
                var expr = parser1.ConsumeAfterTableSource(tableName);
                parser1.ConsumeToken(SqlToken.RParen);
                return new GroupSqlCodeExpr
                {
                    InnerExpr = expr
                };
            }

            return parser.ParseAny(ParseFuncNameIdentifier,
                ParseIdentifierFunc,
                ParseVariable,
                ParseTempTable,
                ParseIdentifier,
                ParseNonIdentifier,
                ParseRentTableWithOption,
                ParseRent);
        }

        private static ToggleSqlCodeExpr ParseToggle(IParser parser)
        {
            if (!parser.TryConsumeTokenAny(out var span,
                    SqlToken.PAD_INDEX, SqlToken.STATISTICS_NORECOMPUTE, SqlToken.IGNORE_DUP_KEY,
                    SqlToken.ALLOW_PAGE_LOCKS, SqlToken.ALLOW_ROW_LOCKS, SqlToken.SORT_IN_TEMPDB,
                    SqlToken.DROP_EXISTING,
                    SqlToken.ONLINE))
            {
                return null;
            }

            var name = parser.Scanner.GetSpanString(span);
            parser.ConsumeToken(SqlToken.Equal);
            var toggleSpan = parser.ConsumeTokenAny(SqlToken.ON, SqlToken.OFF);
            var toggle = parser.Scanner.GetSpanString(toggleSpan);

            return new ToggleSqlCodeExpr
            {
                Name = name.ToUpper(),
                Toggle = toggle
            };
        }

        public static List<SqlCodeExpr> ParseParameterList(this IParser parser, string funcName, int minParameterCount,
            int maxParameterCount)
        {
            var parameterList = new List<SqlCodeExpr>();
            parser.ConsumeToken(SqlToken.LParen);
            do
            {
                if (parser.IsToken(SqlToken.RParen))
                {
                    break;
                }

                if (parameterList.Count >= maxParameterCount)
                {
                    ThrowHelper.ThrowParseException(parser,
                        $"Too many parameters for {funcName}, max is {maxParameterCount}.");
                }

                var parameter = parser.ParseExpIgnoreComment();
                parameterList.Add(parameter);
            } while (parser.MatchToken(SqlToken.Comma));

            parser.ConsumeToken(SqlToken.RParen);

            if (parameterList.Count < minParameterCount)
            {
                ThrowHelper.ThrowParseException(parser,
                    $"{funcName} requires at least {minParameterCount} parameters.");
            }

            return parameterList;
        }
    }

    public class MaxSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("MAX");
        }
    }

    public class AliasSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            LeftExpr.WriteToStream(stream);
            if (AliasName != null)
            {
                stream.Write(" AS ");
                AliasName.WriteToStream(stream);
            }
        }

        public GroupSqlCodeExpr LeftExpr { get; set; }
        public SqlCodeExpr AliasName { get; set; }
    }

    public class AfterActionSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("AFTER ");

            ActionList.Select(x => $"{x}".ToUpper()).WriteToStreamWithComma(stream);
        }

        public List<SqlToken> ActionList { get; set; }
    }

    public class CrossApplySqlCodeExpr : SqlCodeExpr
    {
        public SqlCodeExpr AliasName { get; set; }

        public List<SqlCodeExpr> ColumnList { get; set; }

        public SqlCodeExpr SubExpr { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CROSS APPLY");

            stream.Write(" ");
            SubExpr.WriteToStream(stream);

            if (AliasName != null)
            {
                stream.Write(" AS ");
                AliasName.WriteToStream(stream);
            }

            if (ColumnList != null && ColumnList.Count > 0)
            {
                stream.Write("(");
                ColumnList.WriteToStreamWithComma(stream);
                stream.Write(")");
            }
        }
    }

    public class DataTypeSizeSqlCodeExpr : SqlCodeExpr
    {
        public int? Scale { get; set; }

        public SqlCodeExpr Size { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("(");
            Size.WriteToStream(stream);

            if (Scale != null)
            {
                stream.Write($",{Scale}");
            }

            stream.Write(")");
        }
    }

    public class DefaultSqlCodeExpr : SqlCodeExpr
    {
        public SqlCodeExpr ValueExpr { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("DEFAULT ");
            ValueExpr.WriteToStream(stream);
        }
    }

    public class MarkConstraintSqlCodeExpr : SqlCodeExpr
    {
        public SqlCodeExpr Name { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CONSTRAINT ");
            Name.WriteToStream(stream);
        }
    }

    public class NonClusteredSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("NONCLUSTERED");
        }
    }

    public class NotForReplicationSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("NOT FOR REPLICATION");
        }
    }

    public class NotNullSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("NOT NULL");
        }
    }

    public class OnSqlCodeExpr : SqlCodeExpr
    {
        public SqlCodeExpr Name { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("ON ");
            Name.WriteToStream(stream);
        }
    }

    public class PrimaryKeySqlCodeExpr : SqlCodeExpr
    {
        public List<SqlCodeExpr> ColumnList { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("PRIMARY KEY");
            if (ColumnList != null && ColumnList.Count > 0)
            {
                stream.Write(" (");
                ColumnList.WriteToStreamWithComma(stream);
                stream.Write(")");
            }
        }
    }

    public class ToggleSqlCodeExpr : SqlCodeExpr
    {
        public string Name { get; set; }

        public string Toggle { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($"{Name} = {Toggle.ToUpper()}");
        }
    }

    public class TokenSqlCodeExpr : SqlCodeExpr
    {
        public SqlToken Value { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($"{Value.ToString().ToUpper()}");
        }
    }

    public class TypeObjectIdSqlCodeExpr : SqlCodeExpr
    {
        public SqlCodeExpr ObjectId { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            ObjectId.WriteToStream(stream);
        }
    }

    public class ValueForSqlCodeExpr : SqlCodeExpr
    {
        public SqlCodeExpr Column { get; set; }

        public QuoteStringSqlCodeExpr Value { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            Value.WriteToStream(stream);
            stream.Write(" FOR ");
            Column.WriteToStream(stream);
        }
    }
}