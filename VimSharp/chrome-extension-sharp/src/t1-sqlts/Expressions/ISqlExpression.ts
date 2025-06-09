import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';

export interface ISqlExpression {
    SqlType: SqlType;
    Span: TextSpan;
    Accept(visitor: any): void;
    ToSql(): string;
}

export interface ISqlConstraint extends ISqlExpression {
    ConstraintName: string;
}

export class SqlExpressionNode {
    Expression!: ISqlExpression;
    Depth!: number;
}

// SqlVisitor 只定義基本型別，詳細方法可再擴充
export class SqlVisitor {
    Visit_SqlValue(expr: any): void {}
    Visit_Values(expr: any): void {}
    Visit_NegativeValue(expr: any): void {}
    Visit_NotExpression(expr: any): void {}
    Visit_NullValue(expr: any): void {}
    Visit_OrderByClause(expr: any): void {}
    Visit_OrderColumn(expr: any): void {}
    Visit_OverOrderByClause(expr: any): void {}
    Visit_OverPartitionByClause(expr: any): void {}
    Visit_ParameterValue(expr: any): void {}
    Visit_ParenthesizedExpression(expr: any): void {}
    Visit_PartitionBy(expr: any): void {}
    Visit_RankClause(expr: any): void {}
    Visit_SearchCondition(expr: any): void {}
    Visit_SetValueStatement(expr: any): void {}
    Visit_AddExtendedProperty(expr: any): void {}
    Visit_TableHintIndex(expr: any): void {}
    Visit_TableSource(expr: any): void {}
    Visit_Toggle(expr: any): void {}
    Visit_SqlToken(expr: any): void {}
    Visit_TopClause(expr: any): void {}
    Visit_Distinct(expr: any): void {}
    Visit_ExistsExpression(expr: any): void {}
    Visit_FieldExpr(expr: any): void {}
    Visit_ForXmlAutoClause(expr: any): void {}
    Visit_ForXmlPathClause(expr: any): void {}
    Visit_ForXmlRootDirective(expr: any): void {}
    Visit_FunctionExpression(expr: any): void {}
    Visit_GroupByClause(expr: any): void {}
    Visit_AliasExpr(expr: any): void {}
    Visit_SqlArithmeticBinaryExpr(expr: any): void {}
    Visit_AsExpr(expr: any): void {}
    Visit_CaseClause(expr: any): void {}
    Visit_WhenThen(expr: any): void {}
    Visit_ColumnDefinition(expr: any): void {}
    Visit_DataSize(expr: any): void {}
    Visit_SelectColumn(expr: any): void {}
    Visit_SelectStatement(expr: any): void {}
    Visit_AssignExpr(expr: any): void {}
    Visit_BetweenValue(expr: any): void {}
    Visit_ComparisonOperator(expr: any): void {}
    Visit_ComputedColumnDefinition(expr: any): void {}
    Visit_ConditionExpression(expr: any): void {}
    Visit_ConstraintColumn(expr: any): void {}
    Visit_ConstraintDefaultValue(expr: any): void {}
    Visit_ConstraintForeignKey(expr: any): void {}
    Visit_ConstraintPrimaryKeyOrUnique(expr: any): void {}
    Visit_CreateTableExpression(expr: any): void {}
    Visit_DataTypeWithSize(expr: any): void {}
    Visit_HavingClause(expr: any): void {}
    Visit_Hint(expr: any): void {}
    Visit_InnerTableSource(expr: any): void {}
    Visit_LogicalOperator(expr: any): void {}
    Visit_PivotClause(expr: any): void {}
    Visit_UnaryExpr(expr: any): void {}
    Visit_UnionSelect(expr: any): void {}
    Visit_UnpivotClause(expr: any): void {}
    Visit_JoinTableCondition(expr: any): void {}
    // ... 其他方法依需求擴充
} 