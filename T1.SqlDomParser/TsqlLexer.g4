lexer grammar TsqlLexer;

channels {
	ESQLCOMMENT
}

SPACE: [ \t\r\n]+ -> channel(HIDDEN);
SPEC_ESSQL_COMMENT: '/*!' .+? '*/' -> channel(ESQLCOMMENT);
COMMENT_INPUT: '/*' .*? '*/' -> channel(HIDDEN);
// 遇到 -- 会当作注释跳过 遇到 # 会当作注释跳过
LINE_COMMENT: (
		('-- ' | '#') ~[\r\n]* ('\r'? '\n' | EOF)
		| '--' ('\r'? '\n' | EOF)
	) -> channel(HIDDEN);

DOT: '.';
UNDERLINE: '_';
LBRACKET: '[';
RBRACKET: ']';
LPAREN: '(';
RPAREN: ')';
MINUS: '-';
STAR: '*';
COMMA: ',' | '\uFF0C'; // 使COMMA与,或，等价(\uFF0C表示，的unicode编码)
SEMI: ';';
GT: '>';
SINGLE_QUOTE: '\'';
DOUBLE_QUOTE: '"';
REVERSE_QUOTE: '`';
COLON: ':' | '\uFF1A';
EQ: '=';
NE: '!=';
BOOLOR: '||' | '|'; // 使BOOLOR与||或者|等价
BOOLAND: '&&' | COMMA | '&';

INT: MINUS? DEC_DIGIT+;
DECIMAL: (MINUS? DEC_DIGIT+ DOT DEC_DIGIT+)
	| (MINUS? DOT DEC_DIGIT+);

SELECT: S E L E C T;
ALL: A L L;
DISTINCT: D I S T I N C T;
DELETED: D E L E T E D;
INSERTED: I N S E R T E D;
TOP: T O P;
AS: A S;

ID_LETTER: LETTER+ | LETTER (LETTER | DEC_DIGIT)+;
SQUARE_BRACKET_ID: LBRACKET LETTER+ RBRACKET;
STRING: SINGLE_QUOTE (~'\'')* SINGLE_QUOTE;

fragment DEC_DIGIT: [0-9];
// 使用ID_LETTER代表a-z的大小寫字母和_
fragment LETTER: [a-zA-Z] | UNDERLINE;
fragment A: [aA];
fragment B: [bB];
fragment C: [cC];
fragment D: [dD];
fragment E: [eE];
fragment F: [fF];
fragment G: [gG];
fragment H: [hH];
fragment I: [iI];
fragment J: [jJ];
fragment K: [kK];
fragment L: [lL];
fragment M: [mM];
fragment N: [nN];
fragment O: [oO];
fragment P: [pP];
fragment Q: [qQ];
fragment R: [rR];
fragment S: [sS];
fragment T: [tT];
fragment U: [uU];
fragment V: [vV];
fragment W: [wW];
fragment X: [xX];
fragment Y: [yY];
fragment Z: [zZ];
