namespace WebSample.Services;

public interface IGlobalSettingFactory<out T> 
	where T : new()
{
	T Create();
}