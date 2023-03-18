window.monacoEditorInsertText = function (editorRef, text) {
    const editors = monaco.editor.getEditors().map(x => ({ 
        'id': x.getDomNode().parentNode.id, 
        'editor': x,
    }));
    const editor = editors.filter(x => x.id === editorRef)[0].editor;
    //const editor = window.monaco.editor.getModels()[0];
    const position = editor.getSelection().getPosition();
    //const position = editor.getModel().getPosition();
    editor.executeEdits("insertText", [{
        range: new monaco.Range(position.lineNumber, position.column, position.lineNumber, position.column),
        text: text
    }]);
};

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
    
    const addNewLanguage = (oldLanguage, newLanguage, newCompletionItemProvider) => {
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

        monaco.languages.registerCompletionItemProvider(newLanguage, newCompletionItemProvider);
        
        console.log('create new language', newLanguage)
    };
    
    const newLanguage = 'csharp-' + config.id;
    const newCompletionItemProvider = {
        provideCompletionItems: async function (model, position) {
            const last_chars = model.getValueInRange({
                startLineNumber: position.lineNumber,
                startColumn: 0,
                endLineNumber: position.lineNumber,
                endColumn: position.column
            });
            const words = last_chars.replace("\t", "").split(" ");
            const active_typing = words[words.length - 1];
            
            const t1 = {
                label: 'SELECT',
                kind: monaco.languages.CompletionItemKind.Keyword,
                detail: "Keyword",
                insertText: 'SELECT ',
            };
            return {
                suggestions: [t1]
            };

            //const suggestions = await fetch(config.suggestiosUrl);
            // return {
            //     suggestions: [
            //         {
            //             label: 'SELECT',
            //             kind: monaco.languages.CompletionItemKind.Keyword,
            //             detail: "keyword",
            //             insertText: 'SELECT ',
            //             range: {
            //                 startLineNumber: position.lineNumber,
            //                 endLineNumber: position.lineNumber,
            //                 startColumn: position.column - 6,
            //                 endColumn: position.column,
            //             },
            //         },
            //         {
            //             label: 'FROM',
            //             kind: monaco.languages.CompletionItemKind.Keyword,
            //             insertText: 'FROM ',
            //             range: {
            //                 startLineNumber: position.lineNumber,
            //                 endLineNumber: position.lineNumber,
            //                 startColumn: position.column - 4,
            //                 endColumn: position.column,
            //             },
            //         },
            //     ],
            // };
        },
    };
    
    addNewLanguage('csharp', newLanguage, newCompletionItemProvider);

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