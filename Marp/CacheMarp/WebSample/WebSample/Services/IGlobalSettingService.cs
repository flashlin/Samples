namespace WebSample.Services;

public interface IGlobalSettingService
{
	string GetStringValue(string key);
	bool GetBoolValue(string key);
}