import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlTableHintIndex implements ISqlExpression {
    SqlType: SqlType = SqlType.TableHintIndex;
    Span: TextSpan = new TextSpan();
    IndexValues: ISqlExpression[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_TableHintIndex(this);
    }

    ToSql(): string {
        return this.IndexValues.map(x => x.ToSql()).join(', ');
    }
} 