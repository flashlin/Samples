import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlForXmlRootDirective implements ISqlExpression {
    SqlType: SqlType = SqlType.ForXmlRootDirective;
    Span: TextSpan = new TextSpan();
    RootName?: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ForXmlRootDirective(this);
    }

    ToSql(): string {
        return 'FOR XML ROOT';
    }
} 