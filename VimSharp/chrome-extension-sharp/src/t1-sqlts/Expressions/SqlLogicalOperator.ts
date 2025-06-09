import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { LogicalOperator } from './LogicalOperator';
import { logicalOperatorToSql } from './LogicalOperatorExtensions';

export class SqlLogicalOperator implements ISqlExpression {
    SqlType: SqlType = SqlType.LogicalOperator;
    Span: TextSpan = new TextSpan();
    Value!: LogicalOperator;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_LogicalOperator(this);
    }

    ToSql(): string {
        return logicalOperatorToSql(this.Value);
    }
} 