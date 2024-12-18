namespace T1.SqlSharp.Expressions;

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
    TextSpan Span { get; set; }
    void Accept(SqlVisitor visitor);
}

public class SqlVisitor
{
    public virtual void Visit_SqlArithmeticBinaryExpr(SqlArithmeticBinaryExpr expr)
    {
        expr.Left.Accept(this);
        expr.Right.Accept(this);
    }

    public virtual void Visit_AliasExpr(SqlAliasExpr expr)
    {
    }

    public virtual void Visit_AsExpr(SqlAsExpr expr)
    {
        expr.Instance.Accept(this);
        expr.As.Accept(this);
    }

    public virtual void Visit_ArithmeticBinaryExpr(SqlArithmeticBinaryExpr expr)
    {
        expr.Left.Accept(this);
        expr.Right.Accept(this);
    }

    public virtual void Visit_LogicalOperator(SqlLogicalOperator expr)
    {
    }

    public virtual void Visit_SelectColumn(SelectColumn expr)
    {
        expr.Field.Accept(this);
    }

    public virtual void Visit_SelectStatement(SelectStatement expr)
    {
        expr.Top?.Accept(this);
        expr.Columns.ForEach(x => x.Accept(this));
        expr.FromSources.ForEach(x=>x.Accept(this));
        expr.ForXml?.Accept(this);
        expr.Where?.Accept(this);
        expr.OrderBy?.Accept(this);   
        expr.Unions.ForEach(x=>x.Accept(this));
        expr.GroupBy?.Accept(this);
        expr.Having?.Accept(this);
    }

    public virtual void Visit_SqlToken(SqlToken expr)
    {
    }

    public virtual void Visit_SqlValue(SqlValue expr)
    {
    }

    public virtual void Visit_AssignExpr(SqlAssignExpr expr)
    {
        expr.Left.Accept(this);
        expr.Right.Accept(this);
    }

    public virtual void Visit_BetweenValue(SqlBetweenValue expr)
    {
        expr.Start.Accept(this);
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