namespace PizzaWeb.Models.Helpers;

public interface IJsonConverter
{
    string Serialize<T>(T data);
    T Deserialize<T>(string variablesData);
}