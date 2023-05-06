function getEditor(editorRef) {
    const editors = monaco.editor.getEditors().map(x => ({
        'id': x.getDomNode().parentNode.id,
        'editor': x,
    }));
    return editors.filter(x => x.id === editorRef)[0].editor;
}

function getLineContent(editor) {
    const cursorPosition = editor.getPosition();
    return editor.getModel().getLineContent(cursorPosition.lineNumber);
}

function getPrevLineContent(editor) {
    const cursorPosition = editor.getPosition();
    const model = editor.getModel();
    return model.getValueInRange({
        startLineNumber: cursorPosition.lineNumber,
        startColumn: 0,
        endLineNumber: cursorPosition.lineNumber,
        endColumn: cursorPosition.column
    });
}

function getBlazorInstanceById(id) {
    return window.StaticMonacoEditor.editors[id];
}

window.monacoEditorInsertText = function (editorRef, text) {
    const editor = getEditor(editorRef);
    const position = editor.getSelection().getPosition();
    editor.executeEdits("insertText", [{
        range: new monaco.Range(position.lineNumber, position.column, position.lineNumber, position.column),
        text: text
    }]);
};

window.monacoEditorAppendLine = function (editorRef, text) {
    const editor = getEditor(editorRef);
    const lineCount = editor.getModel().getLineCount();
    const lastLineLength = editor.getModel().getLineMaxColumn(lineCount);
    const range = new monaco.Range(
        lineCount,
        lastLineLength,
        lineCount,
        lastLineLength
    );
    editor.executeEdits('', [
        { range: range, text: text }
    ]);
};

window.StaticMonacoEditor = {
    getPrevContentByCurrentLine: function(editorRef) {
        const editor = getEditor(editorRef);
        const cursorPosition = editor.getPosition();
        const model = editor.getModel();
        return model.getValueInRange({
            startLineNumber: cursorPosition.lineNumber,
            startColumn: 0,
            endLineNumber: cursorPosition.lineNumber,
            endColumn: cursorPosition.column
        });
    },
    getCurrentLineInfo: function (editorRef) {
        const editor = getEditor(editorRef);
        const prev = getPrevLineContent(editor);
        const line = getLineContent(editor);
        const after = line.slice(prev.length);
        const prevText = this.getPrevCursorContent(editorRef);
        return { prev, line, after, prevText };
    },
    getPrevCursorContent: function (editorRef) {
        const editor = getEditor(editorRef);
        const currentPosition = editor.getPosition();
        // // 計算最先開頭的位置
        // let startPosition = new monaco.Position(currentPosition.lineNumber, 1);
        // while (startPosition.lineNumber > 1) {
        //     const lineContent = editor.getModel().getLineContent(startPosition.lineNumber);
        //     if (lineContent.trim() === "") {
        //         startPosition = new monaco.Position(startPosition.lineNumber - 1, 1);
        //     } else {
        //         break;
        //     }
        // }
        const range = new monaco.Range(1, 1, 
            currentPosition.lineNumber, currentPosition.column);
        return editor.getModel().getValueInRange(range);
    }
}

window.monacoEditorIntelliSenseDict = {};

window.monacoEditorSetIntellisense = function (editorRef, list) {
    window.monacoEditorIntelliSenseDict[editorRef] = list;
}

window.monacoEditorTriggerIntelliSense = function (editorRef) {
    const editor = getEditor(editorRef);
    editor.trigger('keyboard', 'editor.action.triggerSuggest', {});
}

function useMonacoEditor1() {
    const myCompletionProvider = {
        provideCompletionItems: function (model, position) {
            const suggestions = [];
            suggestions.push({
                label: 'MyLabel',
                kind: monaco.languages.CompletionItemKind.Keyword,
                insertText: 'MyText',
                range: {
                    startLineNumber: position.lineNumber,
                    endLineNumber: position.lineNumber,
                    startColumn: position.column - 1,
                    endColumn: position.column,
                },
            });
            return {
                suggestions: suggestions,
            };
        },
    };
    monaco.languages.registerCompletionItemProvider('myLanguage', myCompletionProvider);
    const editor = monaco.editor.create(container, {
        value: '',
        language: 'myLanguage',
    });
    editor.setModelLanguage(myModel, 'myLanguage');
}

function createMonacoEditor(config)
{
    const getEditors = () => {
        return monaco.editor.getEditors().map(x => ({
            'id': x.getDomNode().parentNode.id,
            'editor': x,
            'language': x.getModel().getLanguageId()
        }));
    };
    
    const getLanguages = () => {
        return monaco.languages.getLanguages().map(x => x.id);
    }
    
    const addNewLanguage = (oldLanguage, newLanguage, newCompletionItemProvider, options) => {
        if( getLanguages().find(x => x === newLanguage) >= 0 )
        {
            return;
        }
        
        const oldLanguageConfiguration = monaco.languages.getLanguages().filter(function (lang) {
            return lang.id === oldLanguage;
        })[0];
        
        monaco.languages.register({
            id: newLanguage,
            // extensions: oldLanguageConfiguration.extensions,
            // aliases: oldLanguageConfiguration.aliases,
            // mimetypes: oldLanguageConfiguration.mimetypes,
            extensions: ['.cs'],
            aliases: ['C#', 'csharp'],
            mimetypes: ['text/x-csharp'],
        });

        monaco.languages.registerCompletionItemProvider(newLanguage, newCompletionItemProvider, options);
        console.info('create new language', newLanguage)
    };
    
    const newCompletionItemProvider = {
        triggerCharacters: [],
        provideCompletionItems: async function (model, position, context) {
            try {
                // const blazorInstance = getBlazorInstanceById(config.id);
                // const { prev, line, after } = getCurrentLineInfo(blazorInstance.editorRef);
                // const suggestions = await config.dotnetHelper.invokeMethodAsync("MyIntellisense", console.id, {
                //     prevLine: prev,
                //     line: line,
                //     afterLine: after
                // });
                // console.log('completion', prev, suggestions);
                
                // const defaultSuggestions = [{
                //     label: 'SELECT',
                //     kind: monaco.languages.CompletionItemKind.Keyword,
                //     detail: "Keyword",
                //     insertText: 'SELECT ',
                // }];

                // const defaultSuggestions = [];
                const suggestions = window.monacoEditorIntelliSenseDict[config.id] || [];
                // const suggestions = defaultSuggestions.concat(suggestionList);
                //const suggestions = suggestionList;

                return {
                    suggestions
                };
            }catch(e) {
                console.error("completionItemProvider", e);
                return {
                    suggestions: []
                };
            }
        },
    };

    const newLanguage = 'csharp-' + config.id;
    addNewLanguage('csharp', newLanguage, newCompletionItemProvider, 
        {
            triggerCharacters: [],
            //triggerKeyBinding: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Space, monaco.KeyMod.CtrlCmd | monaco.KeyCode.KEY_I]
        });

    const instance = getEditors().filter(x => x.id === config.id)[0];
    const editor = instance.editor;
    if( instance.language === 'csharp') {
        monaco.editor.setModelLanguage(editor.getModel(), newLanguage);
    }
    
    window.StaticMonacoEditor.editors = window.StaticMonacoEditor.editors || {};
    globalEditors = window.StaticMonacoEditor.editors;
    globalEditors[config.id] = { 
        editorRef: config.id, 
        editorInstance: instance
    };
    return this;
}

async function useMonacoEditor(config, dotnetHelper) {
    config.dotnetHelper = dotnetHelper;
    window.monacoEditor = createMonacoEditor(config);
}