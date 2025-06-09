import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlForXmlPathClause implements ISqlExpression {
    SqlType: SqlType = SqlType.ForXmlPathClause;
    Span: TextSpan = new TextSpan();
    CommonDirectives: ISqlExpression[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ForXmlPathClause(this);
    }

    ToSql(): string {
        return 'FOR XML PATH';
    }
} 