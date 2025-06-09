import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlAliasExpr implements ISqlExpression {
    SqlType: SqlType = SqlType.AliasExpr;
    Span: TextSpan = new TextSpan();
    Name: string = '';
    Expression!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_AliasExpr(this);
    }

    ToSql(): string {
        return `${this.Expression.ToSql()} AS ${this.Name}`;
    }
} 