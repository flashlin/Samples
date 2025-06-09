import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { OrderType } from './OrderType';

export class SqlOrderColumn implements ISqlExpression {
    SqlType: SqlType = SqlType.OrderColumn;
    Span: TextSpan = new TextSpan();
    ColumnName!: ISqlExpression;
    Order!: OrderType;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_OrderColumn(this);
    }

    ToSql(): string {
        return `${this.ColumnName.ToSql()} ${this.Order.toString().toUpperCase()}`;
    }
} 