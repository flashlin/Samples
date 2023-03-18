namespace QueryKits.Services;

public class QueryDataSet
{
    public List<Dictionary<string, object>> Rows { get; set; } = new();
}

public class LanguageCompletionItem
{
    public string Label { get; set; } = string.Empty;
    public CompletionItemKind Kind { get; set; }
    public string Detail { get; set; } = string.Empty;
    public string InsertText { get; set; } = string.Empty;
}

public enum CompletionItemKind
{
    Text,  //表示純文字提示
    Method, //表示方法提示
    Function,  //表示函式提示
    Constructor,  //表示建構函式提示
    Field,  //表示欄位提示
    Variable,  //表示變數提示
    Class, //表示類別提示
    Interface,  //表示介面提示
    Module, //表示模組提示
    Property, // 表示屬性提示
    Unit, //表示單位提示
    Value, // 表示值提示
    Enum, // 表示列舉提示
    Keyword,  //表示關鍵字提示
    Snippet,  //表示程式碼片段提示
    Color, // 表示顏色提示
    File, // 表示檔案提示
    Reference, //表示參考提示
    Folder, //表示資料夾提示
    EnumMember, //表示列舉成員提示
    Constant, //表示常數提示
    Struct, //表示結構提示
    Event, //表示事件提示
    Operator, // 表示運算子提示
    TypeParameter, //表示型別參數提示
}

public interface ILanguageService
{
    Task<List<LanguageCompletionItem>> GetIntelliSenseList(string text);
}

public class LanguageService : ILanguageService
{
    public Task<List<LanguageCompletionItem>> GetIntelliSenseList(string text)
    {
        return Task.FromResult(new[]
        {
            new LanguageCompletionItem
            {
                Label = "SELECT",
                Kind = CompletionItemKind.Text,
                Detail = "Keyword",
                InsertText = "SELECT "
            },
        }.ToList());
    }
}