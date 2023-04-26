using OpenAI;
using OpenAI.Chat;
using OpenAI.Completions;
using OpenAI.Models;

public class OpenAiProxy
{
    private string _apiKey;

    public OpenAiProxy(string apiKey)
    {
        _apiKey = apiKey;
    }
    
    public async Task<string> Send(string question)
    {
        var api = new OpenAIClient(_apiKey);
        // var models = await api.ModelsEndpoint.GetModelsAsync();
        // foreach (var model in models)
        // {
        //     Console.WriteLine(model.ToString());
        // }
        
        var message = new List<Message>
        {
            new(Role.System, "You are a helpful assistant."),
            new(Role.User, question),
        };
        var chatRequest = new ChatRequest(message, Model.GPT3_5_Turbo);
        var result = await api.ChatEndpoint.GetCompletionAsync(chatRequest);
        return result.FirstChoice.ToString();
    }
}