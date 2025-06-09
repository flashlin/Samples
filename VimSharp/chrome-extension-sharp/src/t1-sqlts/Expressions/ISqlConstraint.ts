// 對應 C# 的 ISqlConstraint 介面
import { ISqlExpression } from './ISqlExpression';

export interface ISqlConstraint extends ISqlExpression {
    ConstraintName: string;
} 