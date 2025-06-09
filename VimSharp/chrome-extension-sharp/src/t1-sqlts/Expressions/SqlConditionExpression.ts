import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { ComparisonOperator } from './ComparisonOperator';
import { comparisonOperatorToSql } from './ComparisonOperatorExtensions';

export class SqlConditionExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.ComparisonCondition;
    Span: TextSpan = new TextSpan();
    Left!: ISqlExpression;
    ComparisonOperator!: ComparisonOperator;
    OperatorSpan: TextSpan = new TextSpan();
    Right!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ConditionExpression(this);
    }

    ToSql(): string {
        return `${this.Left.ToSql()} ${comparisonOperatorToSql(this.ComparisonOperator)} ${this.Right.ToSql()}`;
    }
} 