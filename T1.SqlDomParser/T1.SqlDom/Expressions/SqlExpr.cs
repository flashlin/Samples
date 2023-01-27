using System.Text;

namespace T1.SqlDom.Expressions
{
    public abstract class SqlExpr
    {
        public static SqlExpr Empty = new EmptySqlExpr();
        public abstract string ToSqlString();

        public virtual List<SqlExpr> ToList()
        {
            return new List<SqlExpr>()
            {
                this,
            };
        }
    }

    public class SqlExprCollection : SqlExpr
    {
        public List<SqlExpr> Items { get; set; } = new List<SqlExpr>();
        
        public override List<SqlExpr> ToList()
        {
            return Items;
        }

        public override string ToSqlString()
        {
            return string.Join(" ", Items.Select(x => x.ToSqlString()));
        }
    }

    public class EmptySqlExpr : SqlExpr
    {
        public override string ToSqlString()
        {
            return string.Empty;
        }
    }
}