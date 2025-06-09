import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlTableSource implements ISqlExpression {
    SqlType: SqlType = SqlType.TableSource;
    Span: TextSpan = new TextSpan();
    Withs: ISqlExpression[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_TableSource(this);
    }

    ToSql(): string {
        return this.Withs.map(x => x.ToSql()).join(', ');
    }
} 