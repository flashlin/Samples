// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "artemis" is now active!');

	// // The command has been defined in the package.json file
	// // Now provide the implementation of the command with registerCommand
	// // The commandId parameter must match the command field in package.json
	// const disposable = vscode.commands.registerCommand('artemis.helloWorld', () => {
	// 	// The code you place here will be executed every time your command is executed
	// 	// Display a message box to the user
	// 	vscode.window.showInformationMessage('Hello World from artemis!');
	// });
	// context.subscriptions.push(disposable);


	let disposable2 = vscode.commands.registerCommand('artemis.provideIntelliSense', async () => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const document = editor.document;
			const documentText = document.getText();
            const position = editor.selection.active;
			const lineText = document.lineAt(position).text;
            const lineStartText = lineText.substring(0, position.character);
            const lineEndText = lineText.substring(position.character);

            // 根據當前行的内容提供建議
            const suggestions = await getSuggestions(lineStartText);

            if (suggestions.length > 0) {
                const selected = await vscode.window.showQuickPick(suggestions, { placeHolder: 'Select an IntelliSense suggestion' });
                if (selected) {
                    editor.edit(editBuilder => {
                        editBuilder.insert(position, selected);
                    });
                }
            }
        }
    });
    context.subscriptions.push(disposable2);
}

async function getSuggestions(lineText: string): Promise<string[]> {
    // 根據行内容提供建議的簡單示例
    if (lineText.includes('hello')) {
        return ['world', 'everyone', 'VS Code'];
    }
    return ['IntelliSense', 'suggestion', 'example'];
}

// This method is called when your extension is deactivated
export function deactivate() {}
