// 對應 C# 的 LinqExpr 相關類別與介面
import { ComparisonOperator } from './ComparisonOperator';
import { LogicalOperator } from './LogicalOperator';

export class LinqExpr {
    From!: LinqFromExpr;
    Where?: LinqWhereExpr;
    OrderBy?: LinqOrderByExpr;
    Joins?: LinqJoinExpr[];
    AdditionalFroms?: LinqFromExpr[];
    Select?: ILinqExpression;
}

export class LinqFromExpr {
    Source!: string;
    AliasName!: string;
    IsDefaultIfEmpty!: boolean;
}

export class LinqSelectAllExpr implements ILinqExpression {
    AliasName!: string;
    equals(obj: any): boolean {
        return obj instanceof LinqSelectAllExpr && obj.AliasName === this.AliasName;
    }
}

export class LinqWhereExpr {
    Condition!: ILinqExpression;
}

export interface ILinqExpression {
    equals(obj: any): boolean;
}

export class LinqConditionExpression implements ILinqExpression {
    Left!: ILinqExpression;
    ComparisonOperator!: ComparisonOperator;
    Right!: ILinqExpression;
    LogicalOperator?: LogicalOperator;
    equals(obj: any): boolean {
        if (!(obj instanceof LinqConditionExpression)) return false;
        if (this.LogicalOperator || obj.LogicalOperator) {
            return this.Left.equals(obj.Left) && this.Right.equals(obj.Right) && this.LogicalOperator === obj.LogicalOperator;
        }
        return this.Left.equals(obj.Left) && this.ComparisonOperator === obj.ComparisonOperator && this.Right.equals(obj.Right);
    }
}

export class LinqFieldExpr implements ILinqExpression {
    TableOrAlias?: string;
    FieldName: string = '';
    equals(obj: any): boolean {
        return obj instanceof LinqFieldExpr && obj.TableOrAlias === this.TableOrAlias && obj.FieldName === this.FieldName;
    }
}

export class LinqValue implements ILinqExpression {
    Value: string = '';
    equals(obj: any): boolean {
        return obj instanceof LinqValue && obj.Value === this.Value;
    }
}

export class LinqOrderByExpr implements ILinqExpression {
    Fields: LinqOrderByFieldExpr[] = [];
    equals(obj: any): boolean {
        return obj instanceof LinqOrderByExpr && JSON.stringify(this.Fields) === JSON.stringify(obj.Fields);
    }
}

export class LinqOrderByFieldExpr implements ILinqExpression {
    Field!: LinqFieldExpr;
    IsDescending!: boolean;
    equals(obj: any): boolean {
        return obj instanceof LinqOrderByFieldExpr && this.Field.equals(obj.Field) && this.IsDescending === obj.IsDescending;
    }
}

export class LinqJoinExpr implements ILinqExpression {
    JoinType: string = 'join';
    AliasName!: string;
    Source!: string;
    On!: LinqConditionExpression;
    Into?: string;
    equals(obj: any): boolean {
        return obj instanceof LinqJoinExpr && this.JoinType === obj.JoinType && this.AliasName === obj.AliasName && this.Source === obj.Source && this.On.equals(obj.On) && this.Into === obj.Into;
    }
}

export class LinqSelectNewExpr implements ILinqExpression {
    Fields!: LinqSelectFieldExpr[];
    equals(obj: any): boolean {
        return obj instanceof LinqSelectNewExpr && JSON.stringify(this.Fields) === JSON.stringify(obj.Fields);
    }
}

export class LinqSelectFieldExpr implements ILinqExpression {
    Name!: string;
    Value!: ILinqExpression;
    equals(obj: any): boolean {
        return obj instanceof LinqSelectFieldExpr && this.Name === obj.Name && this.Value.equals(obj.Value);
    }
} 