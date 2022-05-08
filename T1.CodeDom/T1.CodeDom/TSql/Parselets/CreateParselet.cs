using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            if (parser.IsToken(SqlToken.USER))
            {
                return ParseCreateUser(parser);
            }
            
            if (parser.IsToken(SqlToken.ROLE))
            {
                return ParseCreateRole(parser);
            }
            
            if (parser.IsToken(SqlToken.LOGIN))
            {
                return ParseCreateLogin(parser);
            }
            
            if (parser.TryConsumeToken(out var viewSpan, SqlToken.VIEW))
            {
                return ParseCreateView(viewSpan, parser);
            }

            if (parser.TryConsumeToken(out var typeSpan, SqlToken.TYPE))
            {
                return ParseCreateType(typeSpan, parser);
            }

            if (parser.TryConsumeTokenList(out var spanList, SqlToken.UNIQUE, SqlToken.NONCLUSTERED))
            {
                var createNonClusteredIndexExpr = CreateNonClusteredIndex(spanList[1], parser);
                createNonClusteredIndexExpr.IsUnique = true;
                return createNonClusteredIndexExpr;
            }

            if (parser.TryConsumeToken(out var nonclusteredSpan, SqlToken.NONCLUSTERED))
            {
                return CreateNonClusteredIndex(nonclusteredSpan, parser);
            }

            if (parser.TryConsumeToken(out var synonymSpan, SqlToken.SYNONYM))
            {
                return CreateSynonym(synonymSpan, parser);
            }

            if (parser.Scanner.TryConsume(SqlToken.CLUSTERED, out var clusteredSpan))
            {
                return CreateClusteredIndex(clusteredSpan, parser);
            }

            if (parser.Scanner.TryConsume(SqlToken.Index, out var indexSpan))
            {
                return CreateIndex(indexSpan, parser);
            }

            if (parser.TryConsumeTokenList(out var uniqueSpan, SqlToken.UNIQUE, SqlToken.Index))
            {
                var createIndexExpr = CreateIndex(uniqueSpan[1], parser);
                return new CreateUniqueIndexSqlCodeExpr
                {
                    IndexName = createIndexExpr.IndexName,
                    TableName = createIndexExpr.TableName,
                    OnColumns = createIndexExpr.OnColumns,
                    Comments = createIndexExpr.Comments
                };
            }

            if (parser.Scanner.TryConsume(SqlToken.TABLE, out var tableSpan))
            {
                return parser.CreateTable(tableSpan);
            }

            if (parser.Scanner.Match(SqlToken.Procedure))
            {
                return parser.ConsumeCreateProcedure(token);
            }

            if (parser.Scanner.Match(SqlToken.Function))
            {
                return CreateFunction(token, parser);
            }

            if (parser.Scanner.Match(SqlToken.Partition))
            {
                return CreatePartitionFunction(token, parser);
            }

            if (parser.IsToken(SqlToken.TRIGGER))
            {
                return ConsumeTriggerCreate(parser);
            }

            var helpMessage = parser.Scanner.GetHelpMessage();
            throw new ParseException($"Parse CREATE Error, {helpMessage}");
        }

        private CreateUserSqlCodeExpr ParseCreateUser(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.USER))
            {
                return null;
            }
            
            var userName = parser.ConsumeObjectId();

            SqlCodeExpr loginExpr = null;
            if (parser.MatchTokenList(SqlToken.FOR, SqlToken.LOGIN))
            {
                loginExpr = new ForLoginSqlCodeExpr
                {
                    LoginName = parser.ConsumeObjectId()
                };
            }

            if (parser.MatchTokenList(SqlToken.WITHOUT, SqlToken.LOGIN))
            {
                loginExpr = new WithoutLoginSqlCodeExpr();
            }

            SqlCodeExpr withExpr = null;
            if (parser.MatchTokenList(SqlToken.With, SqlToken.DEFAULT_SCHEMA, SqlToken.Equal))
            {
                var schemaName = parser.ConsumeObjectId();
                withExpr = new WithDefaultSchemaSqlCodeExpr
                {
                    SchemaName = schemaName
                };
            }

            return new CreateUserSqlCodeExpr
            {
                UserName = userName,
                LoginName = loginExpr,
                WithExpr = withExpr 
            };
        }

        private CreateRoleSqlCodeExpr ParseCreateRole(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.ROLE))
            {
                return null;
            }
            var roleName = parser.ConsumeObjectId();
            parser.ConsumeToken(SqlToken.AUTHORIZATION);
            var schemeName = parser.ConsumeObjectId();

            return new CreateRoleSqlCodeExpr
            {
                RoleName = roleName,
                SchemaName = schemeName
            };
        }
            

        private CreateLoginSqlCodeExpr ParseCreateLogin(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.LOGIN))
            {
                return null;
            }

            var loginName = parser.ConsumeObjectId();

            parser.ConsumeToken(SqlToken.With);
            parser.ConsumeToken(SqlToken.PASSWORD);
            parser.ConsumeToken(SqlToken.Equal);

            var password = parser.Consume(SqlToken.QuoteString);

            return new CreateLoginSqlCodeExpr
            {
                LoginName = loginName,
                Password = password
            };
        }

        private SqlCodeExpr ConsumeTriggerCreate(IParser parser)
        {
            var triggerExpr = parser.ConsumeTrigger();

            SqlCodeExpr forTableExpr = null;
            if (parser.MatchToken(SqlToken.FOR))
            {
                forTableExpr = parser.ConsumeObjectId();
            }

            parser.ConsumeToken(SqlToken.As);
            var body = parser.ConsumeBeginBodyOrSingle();
            return new CreateTriggerSqlCodeExpr
            {
                TriggerExpr = triggerExpr,
                ForTableExpr = forTableExpr,
                Body = body
            };
        }

        private IExpression ParseCreateView(TextSpan viewSpan, IParser parser)
        {
            var viewName = parser.ConsumeObjectId();
            parser.ConsumeToken(SqlToken.As);
            var expr = parser.ParseExpIgnoreComment();
            return new CreateViewSqlCodeExpr
            {
                Name = viewName,
                Expr = expr
            };
        }

        private IExpression ParseCreateType(TextSpan typeSpan, IParser parser)
        {
            var typeName = parser.ConsumeObjectId();
            parser.ConsumeToken(SqlToken.As);

            var tableSpan = parser.ConsumeToken(SqlToken.TABLE);
            var typeExpr = parser.ConsumeTableDataTypeList();

            return new CreateTypeSqlCodeExpr
            {
                Name = typeName,
                TypeExpr = typeExpr,
            };
        }

        private CreateNonclusteredIndexSqlCodeExpr CreateNonClusteredIndex(TextSpan nonClusteredSpan, IParser parser)
        {
            parser.ConsumeToken(SqlToken.Index);
            var indexName = parser.ConsumeObjectId();
            parser.ConsumeToken(SqlToken.ON);
            var tableName = parser.ConsumeObjectId();
            parser.ConsumeToken(SqlToken.LParen);
            var columnList = parser.ParseOrderItemList();
            parser.ConsumeToken(SqlToken.RParen);

            SqlCodeExpr whereExpr = null;
            if (parser.MatchToken(SqlToken.Where))
            {
                parser.ConsumeToken(SqlToken.LParen);
                whereExpr = parser.ParseExpIgnoreComment();
                parser.ConsumeToken(SqlToken.RParen);
            }

            var withExpr = parser.ParseConstraintWithOptions();
            var onPrimary = parser.ParseOnPrimary();

            var isSemicolon = parser.MatchToken(SqlToken.Semicolon);
            return new CreateNonclusteredIndexSqlCodeExpr
            {
                IndexName = indexName,
                TableName = tableName,
                ColumnList = columnList,
                WhereExpr = whereExpr,
                WithExpr = withExpr,
                OnPrimary = onPrimary,
                IsSemicolon = isSemicolon,
            };
        }

        private SqlCodeExpr CreateSynonym(TextSpan synonymSpan, IParser parser)
        {
            var synonymName = parser.ConsumeObjectId();
            parser.ConsumeToken(SqlToken.FOR);
            var objectId = parser.ConsumeObjectId();
            return new CreateSynonymSqlCodeExpr
            {
                Name = synonymName,
                ObjectId = objectId
            };
        }

        private SqlCodeExpr CreateClusteredIndex(TextSpan clusteredSpan, IParser parser)
        {
            parser.Scanner.Consume(SqlToken.Index);
            var indexName = parser.ConsumeObjectId();

            parser.Scanner.Consume(SqlToken.ON);
            if (!parser.TryConsumeObjectId(out var tableName))
            {
                tableName = parser.Consume(SqlToken.TempTable);
            }

            parser.Scanner.Consume(SqlToken.LParen);
            var onColumnsList = parser.ParseOrderItemList();
            parser.Scanner.Consume(SqlToken.RParen);

            var withExpr = parser.ParseConstraintWithOptions();

            var onPartitionSchemeNameExpr = ParseOnPartitionSchemeName(parser);

            return new CreateClusteredIndexSqlCodeExpr
            {
                IndexName = indexName,
                TableName = tableName,
                OnColumns = onColumnsList,
                WithExpr = withExpr,
                OnPartitionSchemeNameExpr = onPartitionSchemeNameExpr
            };
        }

        private SqlCodeExpr ParseOnPartitionSchemeName(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.ON))
            {
                return null;
            }

            var filegroupName = parser.ConsumeObjectId();

            if (parser.MatchToken(SqlToken.LParen))
            {
                var column = parser.ConsumeObjectId();
                parser.ConsumeToken(SqlToken.RParen);
                return new OnSqlCodeExpr
                {
                    Name = new PartitionSchemeNameSqlCodeExpr
                    {
                        Name = filegroupName,
                        Column = column
                    }
                };
            }

            return new OnSqlCodeExpr
            {
                Name = filegroupName
            };
        }

        private CreateIndexSqlCodeExpr CreateIndex(TextSpan indexSpan, IParser parser)
        {
            var indexName = parser.ConsumeObjectId();
            parser.Scanner.Consume(SqlToken.ON);

            if (!parser.TryConsumeObjectId(out var tableName))
            {
                tableName = parser.Consume(SqlToken.TempTable);
            }

            var onColumnsList = new List<SqlCodeExpr>();
            parser.Scanner.Consume(SqlToken.LParen);
            do
            {
                var columnName = parser.ConsumeObjectId();
                onColumnsList.Add(columnName);
            } while (parser.Scanner.Match(SqlToken.Comma));

            parser.Scanner.Consume(SqlToken.RParen);

            return new CreateIndexSqlCodeExpr
            {
                IndexName = indexName,
                TableName = tableName,
                OnColumns = onColumnsList
            };
        }

        private IExpression CreatePartitionFunction(TextSpan token, IParser parser)
        {
            if (parser.Scanner.IsToken(SqlToken.Scheme))
            {
                return CreatePartitionScheme(token, parser);
            }

            parser.Scanner.Consume(SqlToken.Function);

            var name = parser.ConsumeObjectId();

            parser.Scanner.Consume(SqlToken.LParen);
            var dataType = parser.ConsumeDataType();
            parser.Scanner.Consume(SqlToken.RParen);

            parser.Scanner.Consume(SqlToken.As);
            parser.Scanner.Consume(SqlToken.Range);

            var rangeType = string.Empty;
            if (parser.Scanner.TryConsumeAny(out var rangeTypeSpan, SqlToken.Left, SqlToken.Right))
            {
                rangeType = parser.Scanner.GetSpanString(rangeTypeSpan);
            }

            parser.Scanner.Consume(SqlToken.FOR);
            parser.Scanner.Consume(SqlToken.Values);

            var boundaryValueList = new List<SqlCodeExpr>();
            parser.Scanner.Consume(SqlToken.LParen);
            do
            {
                var boundaryValue = parser.ParseExpIgnoreComment();
                boundaryValueList.Add(boundaryValue);
            } while (parser.Scanner.Match(SqlToken.Comma));

            parser.Scanner.Consume(SqlToken.RParen);

            return new CreatePartitionFunctionSqlCodeExpr
            {
                Name = name,
                DataType = dataType,
                RangeType = rangeType,
                BoundaryValueList = boundaryValueList,
            };
        }

        private IExpression CreatePartitionScheme(TextSpan token, IParser parser)
        {
            parser.Scanner.Consume(SqlToken.Scheme);
            var schemeName = parser.ConsumeObjectId();

            parser.Scanner.Consume(SqlToken.As);
            parser.Scanner.Consume(SqlToken.Partition);

            var funcName = parser.ConsumeObjectId();

            parser.Scanner.TryConsumeString(SqlToken.ALL, out var allToken);
            parser.Scanner.Consume(SqlToken.To);

            var groupNameList = new List<SqlCodeExpr>();
            parser.Scanner.Consume(SqlToken.LParen);
            do
            {
                groupNameList.Add(parser.ConsumePrimary());
            } while (parser.Scanner.Match(SqlToken.Comma));

            parser.Scanner.Consume(SqlToken.RParen);

            return new CreatePartitionSchemeSqlCodeExpr
            {
                SchemeName = schemeName,
                FuncName = funcName,
                AllToken = allToken,
                GroupNameList = groupNameList
            };
        }

        private IExpression CreateFunction(TextSpan token, IParser parser)
        {
            var nameExpr = parser.ConsumeObjectId();
            parser.Scanner.Consume(SqlToken.LParen);
            var arguments = parser.ConsumeArgumentList();
            parser.Scanner.Consume(SqlToken.RParen);

            parser.Scanner.Consume(SqlToken.Returns);

            parser.Scanner.TryConsumeVariable(out var returnVariableExpr);


            var returnTypeExpr = parser.ConsumeDataType();

            parser.Scanner.Consume(SqlToken.As);

            var body = parser.ConsumeBeginBody();

            return new CreateFunctionSqlCodeExpr
            {
                Name = nameExpr,
                Arguments = arguments,
                ReturnVariable = returnVariableExpr,
                ReturnType = returnTypeExpr,
                Body = body
            };
        }
    }

    public class WithDefaultSchemaSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("WITH DEFAULT_SCHEMA = ");
            SchemaName.WriteToStream(stream);
        }

        public SqlCodeExpr SchemaName { get; set; }
    }

    public class WithoutLoginSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("WITHOUT LOGIN");
        }
    }

    public class ForLoginSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("FOR LOGIN ");
            LoginName.WriteToStream(stream);
        }

        public SqlCodeExpr LoginName { get; set; }
    }

    public class CreateUserSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE USER ");
            UserName.WriteToStream(stream);
            stream.Write(" ");
            LoginName.WriteToStream(stream);
            if (WithExpr != null)
            {
                stream.Write(" ");
                WithExpr.WriteToStream(stream);
            }
        }

        public SqlCodeExpr UserName { get; set; }
        public SqlCodeExpr LoginName { get; set; }
        public SqlCodeExpr WithExpr { get; set; }
    }
}