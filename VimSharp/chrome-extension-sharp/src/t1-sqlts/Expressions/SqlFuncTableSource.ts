import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlFuncTableSource implements ISqlExpression {
    SqlType: SqlType = SqlType.FuncTableSource;
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_FunctionExpression(this);
    }

    ToSql(): string {
        return '';
    }
} 