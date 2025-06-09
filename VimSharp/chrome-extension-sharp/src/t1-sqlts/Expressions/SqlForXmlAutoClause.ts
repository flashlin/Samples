import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlForXmlAutoClause implements ISqlExpression {
    SqlType: SqlType = SqlType.ForXmlAutoClause;
    Span: TextSpan = new TextSpan();
    CommonDirectives: ISqlExpression[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ForXmlAutoClause(this);
    }

    ToSql(): string {
        return 'FOR XML AUTO';
    }
} 