import { ISqlExpression } from './ISqlExpression';
import { ISqlConstraint } from './ISqlConstraint';
import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { SqlVisitor } from './ISqlExpression';

export class SqlCreateTableExpression implements ISqlExpression {
    public Span: TextSpan = new TextSpan();
    public TableName: string = '';
    public Columns: ISqlExpression[] = [];
    public Constraints: ISqlConstraint[] = [];
    
    public get SqlType(): SqlType {
        return SqlType.CreateTable;
    }
    
    public Accept(visitor: SqlVisitor): void {
        visitor.Visit_CreateTableExpression(this);
    }

    public ToSql(): string {
        let sql = `CREATE TABLE ${this.TableName}\n(`;
        sql += this.Columns.map(c => c.ToSql()).join(',\n');
        if (this.Constraints.length > 0) {
            sql += ',\n' + this.Constraints.map(c => c.ToSql()).join(',\n');
        }
        sql += '\n)';
        return sql;
    }
} 