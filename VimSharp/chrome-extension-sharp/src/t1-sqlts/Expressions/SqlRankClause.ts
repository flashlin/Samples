import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export class SqlRankClause implements ISqlExpression {
    SqlType: SqlType = SqlType.RankClause;
    Span: TextSpan = new TextSpan();
    
    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        return 'RANK()';
    }
} 