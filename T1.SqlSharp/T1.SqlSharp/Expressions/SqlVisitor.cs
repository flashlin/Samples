namespace T1.SqlSharp.Expressions;

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
        AddSqlExpression(expr);
        _depth++;
        expr.Left.Accept(this);
        expr.Right.Accept(this);
        _depth--;
    }

    public virtual void Visit_AliasExpr(SqlAliasExpr expr)
    {
    }

    public virtual void Visit_AsExpr(SqlAsExpr expr)
    {
        AddSqlExpression(expr);
        expr.Instance.Accept(this);
        expr.As.Accept(this);
    }

    public virtual void Visit_ArithmeticBinaryExpr(SqlArithmeticBinaryExpr expr)
    {
        AddSqlExpression(expr);
        _depth++;
        expr.Left.Accept(this);
        expr.Right.Accept(this);
        _depth--;
    }

    public virtual void Visit_LogicalOperator(SqlLogicalOperator expr)
    {
    }

    public virtual void Visit_SelectColumn(SelectColumn expr)
    {
        AddSqlExpression(expr);
        expr.Field.Accept(this);
    }

    public virtual void Visit_SelectStatement(SelectStatement expr)
    {
        AddSqlExpression(expr);
        expr.Top?.Accept(this);
        expr.Columns.ForEach(x =>
        {
            x.Accept(this);
        });
        expr.FromSources.ForEach(x=>
        {
            x.Accept(this);
        });
        if(expr.ForXml != null)
        {
            expr.ForXml.Accept(this);
        }
        if(expr.Where != null)
        {
            expr.Where.Accept(this);
        }
        if(expr.OrderBy != null)
        {
            expr.OrderBy.Accept(this);
        }
        expr.Unions.ForEach(x=>
        {
            x.Accept(this);
        });
        if(expr.GroupBy != null)
        {
            expr.GroupBy.Accept(this);
        }
        if(expr.Having != null)
        {
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
        AddSqlExpression(expr);
        expr.Left.Accept(this);
        expr.Right.Accept(this);
    }

    public virtual void Visit_BetweenValue(SqlBetweenValue expr)
    {
        AddSqlExpression(expr);
        expr.Start.Accept(this);
        expr.End.Accept(this);
    }

    public virtual void Visit_CaseClause(SqlCaseClause expr)
    {
        AddSqlExpression(expr);
        expr.Case?.Accept(this);
        expr.WhenThens.ForEach(x=>x.Accept(this));
        expr.Else?.Accept(this);
    }

    public virtual void Visit_WhenThen(SqlWhenThenClause expr)
    {
        AddSqlExpression(expr);
        expr.When.Accept(this);
        expr.Then.Accept(this);
    }

    public virtual void Visit_ColumnDefinition(SqlColumnDefinition expr)
    {
        AddSqlExpression(expr);
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
        AddSqlExpression(expr);
        expr.Left.Accept(this);
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
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ConstraintPrimaryKeyOrUnique(SqlConstraintPrimaryKeyOrUnique expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
        expr.WithToggles.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_CreateTableExpression(SqlCreateTableExpression expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
        expr.Constraints.ForEach(x=>x.Accept(this));
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
        AddSqlExpression(expr);
        expr.Query.Accept(this);
    }

    public virtual void Visit_FieldExpr(SqlFieldExpr expr)
    {
    }

    public virtual void Visit_ForXmlAutoClause(SqlForXmlAutoClause expr)
    {
        AddSqlExpression(expr);
        expr.CommonDirectives.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ForXmlPathClause(SqlForXmlPathClause expr)
    {
        AddSqlExpression(expr);
        expr.CommonDirectives.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ForXmlRootDirective(SqlForXmlRootDirective expr)
    {
        AddSqlExpression(expr);
        expr.RootName?.Accept(this);
    }

    public virtual void Visit_FunctionExpression(SqlFunctionExpression expr)
    {
        AddSqlExpression(expr);
        expr.Parameters.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_Group(SqlGroup expr)
    {
        AddSqlExpression(expr);
        expr.Inner.Accept(this);
    }

    public virtual void Visit_GroupByClause(SqlGroupByClause expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_HavingClause(SqlHavingClause expr)
    {
        AddSqlExpression(expr);
        expr.Condition.Accept(this);
    }

    public virtual void Visit_Hint(SqlHint expr)
    {
    }

    public virtual void Visit_Identity(SqlIdentity expr)
    {
    }

    public virtual void Visit_JoinTableCondition(SqlJoinTableCondition expr)
    {
        AddSqlExpression(expr);
        expr.JoinedTable.Accept(this);
        expr.OnCondition.Accept(this);
    }

    public virtual void Visit_NegativeValue(SqlNegativeValue expr)
    {
        AddSqlExpression(expr);
        expr.Value.Accept(this);
    }

    public virtual void Visit_NotExpression(SqlNotExpression expr)
    {
        AddSqlExpression(expr);
        expr.Value.Accept(this);
    }

    public virtual void Visit_NullValue(SqlNullValue expr)
    {
    }

    public virtual void Visit_OrderByClause(SqlOrderByClause expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_OrderColumn(SqlOrderColumn expr)
    {
        AddSqlExpression(expr);
        expr.ColumnName.Accept(this);
    }

    public virtual void Visit_OverOrderByClause(SqlOverOrderByClause expr)
    {
        AddSqlExpression(expr);
        expr.Field.Accept(this);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_OverPartitionByClause(SqlOverPartitionByClause expr)
    {
        AddSqlExpression(expr);
        expr.Field.Accept(this);
        expr.By.Accept(this);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ParameterValue(SqlParameterValue expr)
    {
    }

    public virtual void Visit_PartitionBy(SqlPartitionBy expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_RankClause(SqlRankClause expr)
    {
        AddSqlExpression(expr);
        expr.PartitionBy?.Accept(this);
        expr.OrderBy.Accept(this);
    }

    public virtual void Visit_SearchCondition(SqlSearchCondition expr)
    {
        AddSqlExpression(expr);
        expr.Left.Accept(this);
        expr.Right?.Accept(this);
    }

    public virtual void Visit_SetValueStatement(SqlSetValueStatement expr)
    {
        AddSqlExpression(expr);
        expr.Name.Accept(this);
        expr.Value.Accept(this);
    }

    public virtual void Visit_AddExtendedProperty(SqlSpAddExtendedPropertyExpression expr)
    {
    }

    public virtual void Visit_TableHintIndex(SqlTableHintIndex expr)
    {
        AddSqlExpression(expr);
        expr.IndexValues.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_TableSource(SqlTableSource expr)
    {
        AddSqlExpression(expr);
        expr.Withs.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_Toggle(SqlToggle expr)
    {
    }

    public virtual void Visit_TopClause(SqlTopClause expr)
    {
        AddSqlExpression(expr);
        expr.Expression.Accept(this);
    }

    public virtual void Visit_UnaryExpr(SqlUnaryExpr expr)
    {
        AddSqlExpression(expr);
        expr.Operand.Accept(this);
    }

    public virtual void Visit_UnionSelect(SqlUnionSelect expr)
    {
        AddSqlExpression(expr);
        expr.SelectStatement.Accept(this);
    }

    public virtual void Visit_UnpivotClause(SqlUnpivotClause expr)
    {
        AddSqlExpression(expr);
        expr.NewColumn.Accept(this);
        expr.ForSource.Accept(this);
        expr.InColumns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_PivotClause(SqlPivotClause expr)
    {
        AddSqlExpression(expr);
        expr.NewColumn.Accept(this);
        expr.ForSource.Accept(this);
        expr.InColumns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_Values(SqlValues expr)
    {
        AddSqlExpression(expr);
        expr.Items.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ChangeTableChanges(SqlChangeTableChanges expr)
    {
    }

    public virtual void Visit_ChangeTableVersion(SqlChangeTableVersion expr)
    {
    }
}