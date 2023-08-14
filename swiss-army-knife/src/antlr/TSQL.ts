import { TSQLLexer } from "@/generated/TSQLLexer";
import { TSQLListener } from "@/generated/TSQLListener";
import { ANTLRInputStream, CommonTokenStream } from 'antlr4ts';

export class TSqlExprListener implements TSQLListener {

}

export class Tsql {
    parse(sql: string) {
        const charStream = new ANTLRInputStream(sql);
        const lexer = new TSQLLexer(charStream);

        const tokenStream = new CommonTokenStream(lexer);
        const parser = new cool_langParser(tokenStream);

        // Walk the syntax tree from the ‘file’ rule as the beginning
        const ruleContext = parser.file();
        ParseTreeWalker.DEFAULT.walk(new CoolLangListener(), ruleContext);
    }
}