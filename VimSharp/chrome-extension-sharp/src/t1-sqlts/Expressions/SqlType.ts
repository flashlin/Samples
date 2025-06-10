import { TextSpan } from "../StringParser";

export enum SqlType {
    TextSpan,
    Field,
    Function,
    Select,
    Value,
    Distinct,
    Table,
    CreateTable,
    AddExtendedProperty,
    SetValue,
    TableSource,
    TableJoin,
    SearchCondition,
    ArithmeticExpr,
    ConditionExpr,
    DistinctExpr,
    NotExpr,
    Null,
    TopClause,
    OrderBy,
    GroupBy,
    Having,
    ChangeTable,
    AliasExpr,
    ExistsExpression,
    Values,
    UnaryExpression,
    NegativeValue,
    WhenThenClause,
    CaseClause,
    RankClause,
    OverPartitionBy,
    OverOrderBy,
    BetweenValue,
    AsExpr,
    ColumnDefinition,
    Constraint,
    ConstraintPK,
    ConstraintFK,
    ConstraintDefault,
    ConstraintCheck,
    Identity,
    SelectStatement,
    IntValue,
    HexValue,
    ParameterValue,
    Token,
    JoinCondition,
    OrderColumn,
    ComparisonCondition,
    Hint,
    Group,
    OrderByClause,
    TableHintIndex,
    FuncTableSource,
    SelectExpression,
    FuncParam,
    PartitionBy,
    UnpivotClause,
    PivotClause,
    ForXmlClause,
    ForXmlPathClause,
    ForXmlAutoClause,
    ForXmlRootDirective,
    UnionSelect,
    AllClause,
    DistinctClause,
    NoneClause
}

export class ParseError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'ParseError';
    }
} 

export class SqlExpr
{
    constructor(sqlType: SqlType, span: TextSpan) {
        this.SqlType = sqlType;
        this.Span = span;
    }
    SqlType: SqlType;
    Span: TextSpan;
}