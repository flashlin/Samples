window.monacoEditorInsertText = function (editorRef, text) {
    const editors = monaco.editor.getEditors().map(x => ({ 
        'id': x.getDomNode().parentNode.id, 
        'editor': x,
    }));
    const editor = editors.filter(x => x.id === editorRef)[0];
    //const editor = window.monaco.editor.getModels()[0];
    const position = editor.getModel().getPosition();
    editor.executeEdits("insertText", [{
        range: new monaco.Range(position.lineNumber, position.column, position.lineNumber, position.column),
        text: text
    }]);
};
