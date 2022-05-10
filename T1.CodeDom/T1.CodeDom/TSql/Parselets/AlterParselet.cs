using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.Common;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class AlterParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            if (parser.TryConsumeToken(out var indexSpan, SqlToken.Index))
            {
                return AlterIndex(indexSpan, parser);
            }
            
            if (parser.TryConsumeToken(out var databaseSpan, SqlToken.DATABASE))
            {
                return AlterDatabase(databaseSpan, parser);
            }

            if (parser.TryConsumeToken(out var tableSpan, SqlToken.TABLE))
            {
                return AlterTable(tableSpan, parser);
            }
            
            if (parser.TryConsumeToken(out var storeProcedureSpan, SqlToken.Procedure))
            {
                return AlterStoreProcedure(storeProcedureSpan, parser);
            }

            var helpMessage = parser.Scanner.GetHelpMessage();
            throw new ParseException(helpMessage);
        }

        private SqlCodeExpr AlterStoreProcedure(TextSpan storeProcedureSpan, IParser parser)
        {
            var createExpr = (CreateProcedureSqlCodeExpr)parser.ConsumeCreateProcedure(storeProcedureSpan);
            return new AlterProcedureSqlCodeExpr
            {
                Name = createExpr.Name,
                Arguments = createExpr.Arguments,
                Body = createExpr.Body,
                Comments = createExpr.Comments,
                WithExecuteAs = createExpr.WithExecuteAs
            };
        }

        private SqlCodeExpr AlterIndex(TextSpan indexSpan, IParser parser)
        {
            var indexName = parser.ParseExpIgnoreComment(int.MaxValue);
            parser.ConsumeToken(SqlToken.ON);
            var tableName = parser.ConsumeObjectId();

            if (parser.MatchToken(SqlToken.REORGANIZE))
            {
                return new AlterIndexReogranizeSqlCodeExpr
                {
                    TableName = tableName
                };
            }
            
            parser.ConsumeToken(SqlToken.Rebuild);
            var withExpr = parser.ParseConstraintWithOptions();
            return new AlterIndexSqlCodeExpr
            {
                IndexName = indexName,
                TableName = tableName,
                WithExpr = withExpr
            };
        }

        private IExpression AlterTable(TextSpan tableSpan, IParser parser)
        {
            var tableName = parser.ConsumeObjectId();

            //var alterAction = parser.ConsumeTokenStringAny(SqlToken.ADD, SqlToken.Set, SqlToken.Drop, SqlToken.NOCHECK, SqlToken.CHECK);

            var alterActions = parser.ConsumeTokenListAny(new[] {SqlToken.ADD},
                new[] {SqlToken.Set},
                new[] {SqlToken.Drop},
                new[] {SqlToken.NOCHECK},
                new []{SqlToken.CHECK },
                new []{SqlToken.With, SqlToken.NOCHECK});

            var alterAction = string.Join(" ", alterActions.Select(x=> parser.Scanner.GetSpanString(x)));

            var optionList = parser.ParseAll(
                ParseLRParenOptionList,
                SqlParserExtension.ParseClustered,
                SqlParserExtension.ParsePrimaryKey,
                SqlParserExtension.ParseConstraint,
                SqlParserExtension.ParseDefault,
                ParseFor,
                ParseColumnDataTypeList
            );

            return new AlterTableSqlCodeExpr
            {
                TableName = tableName,
                AlterAction = alterAction,
                OptionList = optionList
            };
        }

        private ExprListSqlCodeExpr ParseColumnDataTypeList(IParser parser)
        {
            var startIndex = parser.Scanner.GetOffset();
            if (!parser.TryConsumeObjectId(out _))
            {
                return null;
            }

            var dataType = parser.ConsumeTokenString().ToUpper();
            if (!IsDataTypeToken(dataType))
            {
                parser.Scanner.SetOffset(startIndex);
                return null;
            }

            parser.Scanner.SetOffset(startIndex);
            var columnList = new List<SqlCodeExpr>();
            do
            {
                var field = parser.ConsumeObjectId();
                var dataTypeExpr = parser.ConsumeDataType();
                columnList.Add(
                    new ColumnDataTypeSqlCodeExpr
                    {
                        Name = field,
                        DataType = dataTypeExpr
                    });
            } while (parser.MatchToken(SqlToken.Comma));

            return new ExprListSqlCodeExpr
            {
                Items = columnList
            };
        }

        private static bool IsDataTypeToken(string dataType)
        {
            return TSqlParser.DataTypes.Select(x => x.ToString().ToUpper())
                .Contains(dataType);
        }

        private SqlCodeExpr ParseLRParenOptionList(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.LParen))
            {
                return null;
            }

            var optionList = new List<SqlCodeExpr>();
            do
            {
                var option = ParseLockEscalationEq(parser);
                optionList.Add(option);
            } while (parser.MatchToken(SqlToken.Comma));

            parser.ConsumeToken(SqlToken.RParen);

            return new LRParenOptionListSqlCodeExpr
            {
                OptionList = optionList
            };
        }

        private SqlCodeExpr ParseLockEscalationEq(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.LOCK_ESCALATION))
            {
                return null;
            }

            parser.ConsumeToken(SqlToken.Equal);
            var optionValue = parser.ConsumeTokenTypeAny(SqlToken.AUTO, SqlToken.TABLE, SqlToken.DISABLE);
            return new SetOptionSqlCodeExpr
            {
                OptionName = new TokenSqlCodeExpr
                {
                    Value = SqlToken.LOCK_ESCALATION
                },
                OptionValue = new TokenSqlCodeExpr
                {
                    Value = optionValue
                },
            };
        }

        private SqlCodeExpr ParseFor(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.FOR))
            {
                return null;
            }

            var objectId = parser.ConsumeObjectId();
            return new ForSqlCodeExpr
            {
                ObjectId = objectId
            };
        }

        private IExpression AlterDatabase(TextSpan databaseSpan, IParser parser)
        {
            SqlCodeExpr databaseName = null;
            if (parser.MatchToken(SqlToken.CURRENT))
            {
                databaseName = new ObjectIdSqlCodeExpr
                {
                    ObjectName = "CURRENT"
                };
            }
            else
            {
                databaseName = parser.ConsumeObjectId();
            }

            var actionExpr = parser.ConsumeAny(
                ParseSetOnOff,
                ParseAddFileGroup);

            var isSemicolon = parser.MatchToken(SqlToken.Semicolon);

            return new AlterDatabaseSqlCodeExpr
            {
                DatabaseName = databaseName,
                ActionExpr = actionExpr,
                IsSemicolon = isSemicolon,
            };
        }

        private static SqlCodeExpr ParseAddFileGroup(IParser parser)
        {
            if (!parser.MatchTokenList(SqlToken.ADD, SqlToken.FILEGROUP))
            {
                return null;
            }
            var fileGroupName = parser.ConsumeObjectId();
            return new AddFileGroupSqlCodeExpr
            {
                FileGroupName = fileGroupName
            };
        }

        private static SetSqlCodeExpr ParseSetOnOff(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.Set))
            {
                return null;
            }

            var optionName = parser.ConsumeTokenString();
            var toggle = parser.ConsumeTokenStringAny(SqlToken.ON, SqlToken.OFF);

            return new SetSqlCodeExpr
            {
                Options = new[] {optionName}.ToList(),
                Toggle = toggle
            };
        }
    }

    public class AddFileGroupSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("ADD FILEGROUP ");
            FileGroupName.WriteToStream(stream);
        }

        public SqlCodeExpr FileGroupName { get; set; }
    }

    public class AlterIndexReogranizeSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("ALTER INDEX ALL ON ");
            TableName.WriteToStream(stream);
            stream.Write(" REORGANIZE");
        }

        public SqlCodeExpr TableName { get; set; }
    }

    public class AlterIndexSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("ALTER INDEX ");
            IndexName.WriteToStream(stream);
            stream.Write(" ON ");
            TableName.WriteToStream(stream);
            stream.Write(" REBUILD");

            if (WithExpr != null)
            {
                stream.Write(" ");
                WithExpr.WriteToStream(stream);
            }
        }

        public SqlCodeExpr IndexName { get; set; }
        public SqlCodeExpr TableName { get; set; }
        public SqlCodeExpr WithExpr { get; set; }
    }

    public class ColumnDataTypeSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            Name.WriteToStream(stream);
            stream.Write(" ");
            DataType.WriteToStream(stream);
        }

        public SqlCodeExpr Name { get; set; }
        public SqlCodeExpr DataType { get; set; }
    }

    public class LRParenOptionListSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("(");
            OptionList.WriteToStreamWithComma(stream);
            stream.Write(")");
        }

        public List<SqlCodeExpr> OptionList { get; set; }
    }

    public class SetOptionSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            OptionName.WriteToStream(stream);
            stream.Write(" = ");
            OptionValue.WriteToStream(stream);
        }

        public TokenSqlCodeExpr OptionName { get; set; }
        public TokenSqlCodeExpr OptionValue { get; set; }
    }

    public class ForSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("FOR ");
            ObjectId.WriteToStream(stream);
        }

        public SqlCodeExpr ObjectId { get; set; }
    }

    public class AlterTableSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("ALTER TABLE ");
            TableName.WriteToStream(stream);

            stream.Write($" {AlterAction.ToUpper()} ");
            // ConstraintExpr.WriteToStream(stream);
            // stream.Write(" ");
            // DefaultExpr.WriteToStream(stream);
            // stream.Write(" FOR ");
            // ForExpr.WriteToStream(stream);
            OptionList.WriteToStream(stream);
        }

        public SqlCodeExpr TableName { get; set; }

        // public MarkConstraintSqlCodeExpr ConstraintExpr { get; set; }
        // public SqlCodeExpr DefaultExpr { get; set; }
        // public SqlCodeExpr ForExpr { get; set; }
        public List<SqlCodeExpr> OptionList { get; set; }
        public string AlterAction { get; set; }
    }
}