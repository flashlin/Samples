import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { ComparisonOperator } from './ComparisonOperator';
import { comparisonOperatorToSql } from './ComparisonOperatorExtensions';

export class SqlComparisonOperator implements ISqlExpression {
    SqlType: SqlType = SqlType.ComparisonOperator;
    Span: TextSpan = new TextSpan();
    Value!: ComparisonOperator;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ComparisonOperator(this);
    }

    ToSql(): string {
        return comparisonOperatorToSql(this.Value);
    }
} 