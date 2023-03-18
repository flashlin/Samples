namespace QueryKits.Services;

public interface IJsJsonSerializer
{
    string Serialize(object obj);
    T? Deserialize<T>(string json);
}