import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export class SqlChangeTableChanges implements ISqlExpression {
    SqlType: SqlType = SqlType.ChangeTable;
    Span: TextSpan = new TextSpan();
    TableName: string = '';
    LastSyncVersion!: ISqlExpression;
    Alias: string = '';

    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        return `CHANGETABLE(CHANGES ${this.TableName}, ${this.LastSyncVersion.ToSql()})${this.Alias ? ' AS ' + this.Alias : ''}`;
    }
} 