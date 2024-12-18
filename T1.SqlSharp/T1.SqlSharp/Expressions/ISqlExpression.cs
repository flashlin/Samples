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
    }

    public virtual void Visit_AliasExpr(SqlAliasExpr sqlAliasExpr)
    {
    }

    public virtual void Visit_AsExpr(SqlAsExpr sqlAsExpr)
    {
    }

    public virtual void Visit_ArithmeticBinaryExpr(SqlArithmeticBinaryExpr sqlArithmeticBinaryExpr)
    {
    }

    public virtual void Visit_LogicalOperator(SqlLogicalOperator sqlLogicalOperator)
    {
    }

    public virtual void Visit_SelectColumn(SelectColumn selectColumn)
    {
    }

    public virtual void Visit_SelectStatement(SelectStatement selectStatement)
    {
    }

    public virtual void Visit_SqlToken(SqlToken sqlToken)
    {
    }

    public virtual void Visit_SqlValue(SqlValue sqlValue)
    {
    }

    public virtual void Visit_AssignExpr(SqlAssignExpr sqlAssignExpr)
    {
    }

    public virtual void Visit_BetweenValue(SqlBetweenValue sqlBetweenValue)
    {
    }

    public virtual void Visit_CaseClause(SqlCaseClause sqlCaseClause)
    {
    }

    public virtual void Visit_WhenThen(SqlWhenThenClause sqlWhenThenClause)
    {
    }

    public virtual void Visit_ColumnDefinition(SqlColumnDefinition sqlColumnDefinition)
    {
    }

    public virtual void Visit_ComparisonOperator(SqlComparisonOperator sqlComparisonOperator)
    {
    }

    public virtual void Visit_ComputedColumnDefinition(SqlComputedColumnDefinition sqlComputedColumnDefinition)
    {
    }

    public virtual void Visit_ConditionExpression(SqlConditionExpression sqlConditionExpression)
    {
    }

    public virtual void Visit_ConstraintColumn(SqlConstraintColumn sqlConstraintColumn)
    {
    }

    public virtual void Visit_ConstraintDefaultValue(SqlConstraintDefaultValue sqlConstraintDefaultValue)
    {
    }

    public virtual void Visit_ConstraintForeignKey(SqlConstraintForeignKey sqlConstraintForeignKey)
    {
    }

    public virtual void Visit_ConstraintPrimaryKeyOrUnique(SqlConstraintPrimaryKeyOrUnique sqlConstraintPrimaryKeyOrUnique)
    {
    }

    public virtual void Visit_CreateTableExpression(SqlCreateTableExpression sqlCreateTableExpression)
    {
    }

    public virtual void Visit_DataSize(SqlDataSize sqlDataSize)
    {
    }

    public virtual void Visit_DataTypeWithSize(SqlDataTypeWithSize sqlDataTypeWithSize)
    {
    }

    public virtual void Visit_Distinct(SqlDistinct sqlDistinct)
    {
    }

    public virtual void Visit_ExistsExpression(SqlExistsExpression sqlExistsExpression)
    {
    }

    public virtual void Visit_FieldExpr(SqlFieldExpr sqlFieldExpr)
    {
    }

    public virtual void Visit_ForXmlAutoClause(SqlForXmlAutoClause sqlForXmlAutoClause)
    {
    }

    public virtual void Visit_ForXmlPathClause(SqlForXmlPathClause sqlForXmlPathClause)
    {
    }

    public virtual void Visit_ForXmlRootDirective(SqlForXmlRootDirective sqlForXmlRootDirective)
    {
    }

    public virtual void Visit_FunctionExpression(SqlFunctionExpression sqlFunctionExpression)
    {
    }

    public virtual void Visit_Group(SqlGroup sqlGroup)
    {
    }

    public virtual void Visit_GroupByClause(SqlGroupByClause sqlGroupByClause)
    {
    }

    public virtual void Visit_HavingClause(SqlHavingClause sqlHavingClause)
    {
    }

    public virtual void Visit_Hint(SqlHint sqlHint)
    {
    }

    public virtual void Visit_Identity(SqlIdentity sqlIdentity)
    {
    }

    public virtual void Visit_JoinTableCondition(SqlJoinTableCondition sqlJoinTableCondition)
    {
    }

    public virtual void Visit_NegativeValue(SqlNegativeValue sqlNegativeValue)
    {
    }

    public virtual void Visit_NotExpression(SqlNotExpression sqlNotExpression)
    {
    }

    public virtual void Visit_NullValue(SqlNullValue sqlNullValue)
    {
    }

    public virtual void Visit_OrderByClause(SqlOrderByClause sqlOrderByClause)
    {
    }

    public virtual void Visit_OrderColumn(SqlOrderColumn sqlOrderColumn)
    {
    }

    public virtual void Visit_OverOrderByClause(SqlOverOrderByClause sqlOverOrderByClause)
    {
    }

    public virtual void Visit_OverPartitionByClause(SqlOverPartitionByClause sqlOverPartitionByClause)
    {
    }

    public virtual void Visit_ParameterValue(SqlParameterValue sqlParameterValue)
    {
    }

    public virtual void Visit_PartitionBy(SqlPartitionBy sqlPartitionBy)
    {
    }

    public virtual void Visit_RankClause(SqlRankClause sqlRankClause)
    {
    }

    public virtual void Visit_SearchCondition(SqlSearchCondition sqlSearchCondition)
    {
    }

    public virtual void Visit_SetValueStatement(SqlSetValueStatement sqlSetValueStatement)
    {
    }

    public virtual void Visit_AddExtendedProperty(SqlSpAddExtendedPropertyExpression sqlSpAddExtendedPropertyExpression)
    {
    }

    public virtual void Visit_TableHintIndex(SqlTableHintIndex sqlTableHintIndex)
    {
    }

    public virtual void Visit_TableSource(SqlTableSource sqlTableSource)
    {
    }

    public virtual void Visit_Toggle(SqlToggle sqlToggle)
    {
    }

    public virtual void Visit_TopClause(SqlTopClause sqlTopClause)
    {
    }

    public virtual void Visit_UnaryExpr(SqlUnaryExpr sqlUnaryExpr)
    {
    }

    public virtual void Visit_UnionSelect(SqlUnionSelect sqlUnionSelect)
    {
    }

    public virtual void Visit_UnpivotClause(SqlUnpivotClause sqlUnpivotClause)
    {
    }

    public virtual void Visit_PivotClause(SqlPivotClause sqlPivotClause)
    {
    }

    public virtual void Visit_Values(SqlValues sqlValues)
    {
    }
}