import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export class SqlFunctionExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.Function;
    Span: TextSpan = new TextSpan();
    Name: string = '';
    Parameters: ISqlExpression[] = [];
    
    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        return `${this.Name}(${this.Parameters.map(p => p.ToSql()).join(', ')})`;
    }
} 