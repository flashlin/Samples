// 對應 C# 的 ISelectColumnExpression 介面
import { ISqlExpression } from './ISqlExpression';

export interface ISelectColumnExpression extends ISqlExpression {
    Field: ISqlExpression;
    Alias: string;
} 