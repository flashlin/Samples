using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

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

            throw new ParseException();
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
}