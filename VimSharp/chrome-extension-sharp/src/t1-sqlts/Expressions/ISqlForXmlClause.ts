// 對應 C# 的 ISqlForXmlClause 介面
import { ISqlExpression } from './ISqlExpression';
import { SqlForXmlRootDirective } from './SqlForXmlRootDirective';

export interface ISqlForXmlClause extends ISqlExpression {
    CommonDirectives: SqlForXmlRootDirective[];
} 