import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlFunctionExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.Function;
    Span: TextSpan = new TextSpan();
    Parameters: ISqlExpression[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_FunctionExpression(this);
    }

    ToSql(): string {
        return '';
    }
} 