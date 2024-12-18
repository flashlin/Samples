namespace T1.SqlSharp.Expressions;

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
    TextSpan Span { get; set; }
    void Accept(SqlVisitor visitor);
}

public class SqlExpressionNode
{
    public required ISqlExpression Expression { get; set; }
    public int Depth { get; set; }
}

public class SqlVisitor
{
    private readonly List<SqlExpressionNode> _expressions = new();
    private int _depth = 0;
    
    public List<SqlExpressionNode> Visit(ISqlExpression expression)
    {
        _expressions.Clear();
        AddSqlExpression(expression);
        expression.Accept(this);
        return _expressions;
    }

    private void AddSqlExpression(ISqlExpression expression)
    {
        _expressions.Add(new SqlExpressionNode()
        {
            Expression = expression,
            Depth = _depth
        });
    }

    public virtual void Visit_SqlArithmeticBinaryExpr(SqlArithmeticBinaryExpr expr)
    {
        _depth++;
        AddSqlExpression(expr.Left);
        expr.Left.Accept(this);
        AddSqlExpression(expr.Right);
        expr.Right.Accept(this);
        _depth--;
    }

    public virtual void Visit_AliasExpr(SqlAliasExpr expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AsExpr(SqlAsExpr expr)
    {
        AddSqlExpression(expr.Instance);
        expr.Instance.Accept(this);
        AddSqlExpression(expr.As);
        expr.As.Accept(this);
    }

    public virtual void Visit_ArithmeticBinaryExpr(SqlArithmeticBinaryExpr expr)
    {
        _depth++;
        AddSqlExpression(expr.Left);
        expr.Left.Accept(this);
        AddSqlExpression(expr.Right);
        expr.Right.Accept(this);
        _depth--;
    }

    public virtual void Visit_LogicalOperator(SqlLogicalOperator expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_SelectColumn(SelectColumn expr)
    {
        AddSqlExpression(expr.Field);
        expr.Field.Accept(this);
    }

    public virtual void Visit_SelectStatement(SelectStatement expr)
    {
        if (expr.Top != null)
        {
            AddSqlExpression(expr.Top);
        }
        expr.Top?.Accept(this);
        expr.Columns.ForEach(x =>
        {
            AddSqlExpression(x);
            x.Accept(this);
        });
        expr.FromSources.ForEach(x=>
        {
            AddSqlExpression(x);
            x.Accept(this);
        });
        if(expr.ForXml != null)
        {
            AddSqlExpression(expr.ForXml);
            expr.ForXml.Accept(this);
        }
        if(expr.Where != null)
        {
            AddSqlExpression(expr.Where);
            expr.Where.Accept(this);
        }
        if(expr.OrderBy != null)
        {
            AddSqlExpression(expr.OrderBy);
            expr.OrderBy.Accept(this);
        }
        expr.Unions.ForEach(x=>
        {
            AddSqlExpression(x);
            x.Accept(this);
        });
        if(expr.GroupBy != null)
        {
            AddSqlExpression(expr.GroupBy);
            expr.GroupBy.Accept(this);
        }
        if(expr.Having != null)
        {
            AddSqlExpression(expr.Having);
            expr.Having.Accept(this);
        }
    }

    public virtual void Visit_SqlToken(SqlToken expr)
    {
    }

    public virtual void Visit_SqlValue(SqlValue expr)
    {
    }

    public virtual void Visit_AssignExpr(SqlAssignExpr expr)
    {
        AddSqlExpression(expr.Left);
        expr.Left.Accept(this);
        AddSqlExpression(expr.Right);
        expr.Right.Accept(this);
    }

    public virtual void Visit_BetweenValue(SqlBetweenValue expr)
    {
        AddSqlExpression(expr.Start);
        expr.Start.Accept(this);
        AddSqlExpression(expr.End);
        expr.End.Accept(this);
    }

    public virtual void Visit_CaseClause(SqlCaseClause expr)
    {
        expr.Case?.Accept(this);
        expr.WhenThens.ForEach(x=>x.Accept(this));
        expr.Else?.Accept(this);
    }

    public virtual void Visit_WhenThen(SqlWhenThenClause expr)
    {
        expr.When.Accept(this);
        expr.Then.Accept(this);
    }

    public virtual void Visit_ColumnDefinition(SqlColumnDefinition expr)
    {
        expr.Constraints.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ComparisonOperator(SqlComparisonOperator expr)
    {
    }

    public virtual void Visit_ComputedColumnDefinition(SqlComputedColumnDefinition expr)
    {
    }

    public virtual void Visit_ConditionExpression(SqlConditionExpression expr)
    {
        AddSqlExpression(expr.Left);
        expr.Left.Accept(this);
        AddSqlExpression(expr.Right);
        expr.Right.Accept(this);
    }

    public virtual void Visit_ConstraintColumn(SqlConstraintColumn expr)
    {
    }

    public virtual void Visit_ConstraintDefaultValue(SqlConstraintDefaultValue expr)
    {
    }

    public virtual void Visit_ConstraintForeignKey(SqlConstraintForeignKey expr)
    {
    }

    public virtual void Visit_ConstraintPrimaryKeyOrUnique(SqlConstraintPrimaryKeyOrUnique expr)
    {
    }

    public virtual void Visit_CreateTableExpression(SqlCreateTableExpression expr)
    {
    }

    public virtual void Visit_DataSize(SqlDataSize expr)
    {
    }

    public virtual void Visit_DataTypeWithSize(SqlDataTypeWithSize expr)
    {
    }

    public virtual void Visit_Distinct(SqlDistinct expr)
    {
    }

    public virtual void Visit_ExistsExpression(SqlExistsExpression expr)
    {
    }

    public virtual void Visit_FieldExpr(SqlFieldExpr expr)
    {
    }

    public virtual void Visit_ForXmlAutoClause(SqlForXmlAutoClause expr)
    {
    }

    public virtual void Visit_ForXmlPathClause(SqlForXmlPathClause expr)
    {
    }

    public virtual void Visit_ForXmlRootDirective(SqlForXmlRootDirective expr)
    {
    }

    public virtual void Visit_FunctionExpression(SqlFunctionExpression expr)
    {
    }

    public virtual void Visit_Group(SqlGroup expr)
    {
    }

    public virtual void Visit_GroupByClause(SqlGroupByClause expr)
    {
    }

    public virtual void Visit_HavingClause(SqlHavingClause expr)
    {
    }

    public virtual void Visit_Hint(SqlHint expr)
    {
    }

    public virtual void Visit_Identity(SqlIdentity expr)
    {
    }

    public virtual void Visit_JoinTableCondition(SqlJoinTableCondition expr)
    {
    }

    public virtual void Visit_NegativeValue(SqlNegativeValue expr)
    {
    }

    public virtual void Visit_NotExpression(SqlNotExpression expr)
    {
    }

    public virtual void Visit_NullValue(SqlNullValue expr)
    {
    }

    public virtual void Visit_OrderByClause(SqlOrderByClause expr)
    {
    }

    public virtual void Visit_OrderColumn(SqlOrderColumn expr)
    {
    }

    public virtual void Visit_OverOrderByClause(SqlOverOrderByClause expr)
    {
    }

    public virtual void Visit_OverPartitionByClause(SqlOverPartitionByClause expr)
    {
    }

    public virtual void Visit_ParameterValue(SqlParameterValue expr)
    {
    }

    public virtual void Visit_PartitionBy(SqlPartitionBy expr)
    {
    }

    public virtual void Visit_RankClause(SqlRankClause expr)
    {
    }

    public virtual void Visit_SearchCondition(SqlSearchCondition expr)
    {
    }

    public virtual void Visit_SetValueStatement(SqlSetValueStatement expr)
    {
    }

    public virtual void Visit_AddExtendedProperty(SqlSpAddExtendedPropertyExpression expr)
    {
    }

    public virtual void Visit_TableHintIndex(SqlTableHintIndex expr)
    {
    }

    public virtual void Visit_TableSource(SqlTableSource expr)
    {
    }

    public virtual void Visit_Toggle(SqlToggle expr)
    {
    }

    public virtual void Visit_TopClause(SqlTopClause expr)
    {
    }

    public virtual void Visit_UnaryExpr(SqlUnaryExpr expr)
    {
    }

    public virtual void Visit_UnionSelect(SqlUnionSelect expr)
    {
    }

    public virtual void Visit_UnpivotClause(SqlUnpivotClause expr)
    {
    }

    public virtual void Visit_PivotClause(SqlPivotClause expr)
    {
    }

    public virtual void Visit_Values(SqlValues expr)
    {
    }
}