function getEditor(editorRef) {
    const editors = monaco.editor.getEditors().map(x => ({
        'id': x.getDomNode().parentNode.id,
        'editor': x,
    }));
    return editors.filter(x => x.id === editorRef)[0].editor;
}

window.monacoEditorInsertText = function (editorRef, text) {
    const editor = getEditor(editorRef);
    const position = editor.getSelection().getPosition();
    //const position = editor.getModel().getPosition();
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
    getContentByCurrentLine: function (editorRef) {
        const editor = getEditor(editorRef);
        const cursorPosition = editor.getPosition();
        return editor.getModel().getLineContent(cursorPosition.lineNumber);
    }
}

window.monacoEditorIntelliSenseDict = {};

window.monacoEditorSetIntellisense = function (editorRef, list) {
    window.monacoEditorIntelliSenseDict[editorRef] = list;
}

window.monacoEditorTriggerIntelliSense = function (editorRef) {
    const editor = getEditor(editorRef);
    // editor.trigger('editor.action.triggerSuggest', {
    //     source: 'keyboard',
    //     explicit: true
    // });
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
    config = JSON.parse(config);
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
        console.log('create new language', newLanguage)
    };
    
    const newLanguage = 'csharp-' + config.id;
    const newCompletionItemProvider = {
        provideCompletionItems: async function (model, position) {
            try {
                // const last_chars = model.getValueInRange({
                //     startLineNumber: position.lineNumber,
                //     startColumn: 0,
                //     endLineNumber: position.lineNumber,
                //     endColumn: position.column
                // });
                // console.log("last_chars", last_chars);
                // const words = last_chars.replace("\t", "").split(" ");
                // const active_typing = words[words.length - 1];

                // const defaultSuggestions = [{
                //     label: 'SELECT',
                //     kind: monaco.languages.CompletionItemKind.Keyword,
                //     detail: "Keyword",
                //     insertText: 'SELECT ',
                // }];

                const suggestionList = window.monacoEditorIntelliSenseDict[config.id];
                //const suggestions = defaultSuggestions.concat(suggestionList);

                return {
                    suggestions: suggestionList
                };
            }catch(e) {
                console.error("completionItemProvider", e);
                return {
                    suggestions: []
                };
            }
        },
    };
    
    addNewLanguage('csharp', newLanguage, newCompletionItemProvider, 
        {
            //triggerKeyBinding: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Space, monaco.KeyMod.CtrlCmd | monaco.KeyCode.KEY_I]
        });

    const instance = getEditors().filter(x => x.id === config.id)[0];
    const editor = instance.editor;
    if( instance.language === 'csharp') {
        monaco.editor.setModelLanguage(editor.getModel(), newLanguage);
    }

    return this;
}

function useMonacoEditor(config)
{
    window.monacoEditor = createMonacoEditor(config);
}