import * as vscode from 'vscode';

export class EditorEx {
    editor: vscode.TextEditor;
    constructor(editor: vscode.TextEditor) {
        this.editor = editor;
    }

    insertTextAtNextLine(multiLineText: string) {
        const editor = this.editor;
        const originalPosition = editor.selection.active;
        const document = editor.document;
        const position = editor.selection.active;
        const currentLine = document.lineAt(position.line);

        const newPosition = new vscode.Position(currentLine.lineNumber + 1, 0);
        editor.edit(editBuilder => {
            editBuilder.insert(newPosition, multiLineText + '\n');
        }).then(success => {
            if (success) {
                editor.selection = new vscode.Selection(originalPosition, originalPosition);
            }
        });
    }

    /*
    const vscode = acquireVsCodeApi();
    function sendMessage() {
        const text = document.getElementById('multitext').value;
        vscode.postMessage({
            command: 'alert',
            text: text
        });
    }
    */
    outputToPreview(oldPanel: vscode.WebviewPanel | undefined, identify: string, title: string, htmlText: string): vscode.WebviewPanel {
        if( oldPanel !== undefined) {
            oldPanel.webview.html = htmlText;
            return oldPanel;
        }
        const panel = vscode.window.createWebviewPanel(
            identify, // Identifies the type of the webview. Used internally
            title, // Title of the panel displayed to the user
            vscode.ViewColumn.One, // Editor column to show the new webview panel in.
            {
                enableScripts: true // Enable scripts in the webview
            }
        );
        panel.webview.html = htmlText;
        // Handle messages from the webview
        panel.webview.onDidReceiveMessage(
            message => {
                switch (message.command) {
                    case 'alert':
                        vscode.window.showInformationMessage(message.text);
                        return;
                }
            },
        );
        // const activeEditor = vscode.window.activeTextEditor;
        // if( activeEditor ) {
        //     vscode.window.showTextDocument(activeEditor.document, panel.viewColumn, false);
        // }
        return panel;
    }

    showPanel(customPreviewPanel: vscode.WebviewPanel | undefined){
        if( customPreviewPanel ) {
            customPreviewPanel.reveal(vscode.ViewColumn.Two);
        }
    }
}