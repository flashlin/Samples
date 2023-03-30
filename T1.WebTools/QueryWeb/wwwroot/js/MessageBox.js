window.messageBox = {
    show: function (message) {
        return confirm(message);
    }
};

window.copyTextToClipboard = async (text) => {
    await navigator.clipboard.writeText(text);
    alert('copy to clipboard ok');
};
