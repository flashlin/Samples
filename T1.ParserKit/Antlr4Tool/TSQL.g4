grammar TSQL;

// 起始規則
start : selectStatement ;

// SELECT查詢語句的規則
selectStatement : SELECT selectColumnList FROM fromClause ;

// SELECT和FROM的規則，使用大寫不敏感的方式
SELECT : [sS] [eE] [lL] [eE] [cC] [tT] ;
FROM : [fF] [rR] [oO] [mM] ;
AS : [aA] [sS] ;

// SELECT列的規則
selectColumnList : selectColumn ( ',' selectColumn )* ;
selectColumn : ID (AS? ID)? ;

// FROM子句的規則
fromClause : tableReference | '(' selectStatement ')' ;
tableReference : ID (AS? ID)?;

// 忽略空白
WS : [ \t\r\n]+ -> skip ;

// 保留字和ID定義，使用大寫不敏感的方式
ID : [a-zA-Z_] [a-zA-Z_0-9]* ;
