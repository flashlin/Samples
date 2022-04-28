using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class AlterParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
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

        private IExpression AlterTable(TextSpan tableSpan, IParser parser)
        {
            var tableName = parser.ConsumeObjectId();
            
            parser.ConsumeToken(SqlToken.ADD);


            var optionList = parser.ParseAll(
                SqlParserExtension.ParseConstraint,
                SqlParserExtension.ParseDefault,
                ParseFor
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
                // ConstraintExpr = constraintExpr,
                // DefaultExpr = defaultValueExpr,
                // ForExpr = forExpr,
                OptionList = optionList
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
            SqlCodeExpr databaseName  = null;
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
            stream.Write(" ADD ");
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
    }
}