using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
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

            var helpMessage = parser.Scanner.GetHelpMessage();
            throw new ParseException(helpMessage);
        }

        private SqlCodeExpr AlterIndex(TextSpan indexSpan, IParser parser)
        {
            parser.ConsumeToken(SqlToken.ALL);
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
                TableName = tableName,
                WithExpr = withExpr
            };
        }

        private IExpression AlterTable(TextSpan tableSpan, IParser parser)
        {
            var tableName = parser.ConsumeObjectId();

            var alterActionSpan = parser.ConsumeTokenAny(SqlToken.ADD, SqlToken.Set, SqlToken.Drop);
            var alterAction = parser.Scanner.GetSpanString(alterActionSpan);

            var optionList = parser.ParseAll(
                ParseLRParenOptionList,
                SqlParserExtension.ParseClustered,
                SqlParserExtension.ParsePrimaryKey,
                SqlParserExtension.ParseConstraint,
                SqlParserExtension.ParseDefault,
                ParseFor,
                ParseColumnDataTypeList
            );

            // var constraintExpr = parser.ParseConstraint();
            // var defaultValueExpr = parser.ParseDefault();
            // SqlCodeExpr forExpr = null;
            // if (defaultValueExpr != null)
            // {
            //     parser.ConsumeToken(SqlToken.FOR);
            //     forExpr = parser.ConsumeObjectId();
            // }

            return new AlterTableSqlCodeExpr
            {
                TableName = tableName,
                AlterAction = alterAction,
                // ConstraintExpr = constraintExpr,
                // DefaultExpr = defaultValueExpr,
                // ForExpr = forExpr,
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

            parser.ConsumeToken(SqlToken.ADD);
            parser.ConsumeToken(SqlToken.FILEGROUP);

            var filegroupName = parser.ConsumeObjectId();
            var isSemicolon = parser.MatchToken(SqlToken.Semicolon);

            return new AlterDatabaseSqlCodeExpr
            {
                DatabaseName = databaseName,
                FileGroupName = filegroupName,
                IsSemicolon = isSemicolon,
            };
        }
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
            stream.Write("ALTER INDEX ALL ON ");
            TableName.WriteToStream(stream);
            stream.Write(" REBUILD ");
            WithExpr.WriteToStream(stream);
        }

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