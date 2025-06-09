// 對應 C# 的 ITableSource 介面
import { ISqlExpression } from './ISqlExpression';

export interface ITableSource extends ISqlExpression {
    Alias: string;
    Withs: ISqlExpression[];
} 