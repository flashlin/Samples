using System.Text;
using System.Text.Json;

namespace WebSample.Services;

public class GlobalSettingFactory<T> : IGlobalSettingFactory<T> 
	where T : new()
{
	private readonly IGlobalSettingRepo _globalSettingRepo;

	public GlobalSettingFactory(IGlobalSettingRepo globalSettingRepo)
	{
		_globalSettingRepo = globalSettingRepo;
	}

	public T Create()
	{
		var json = new StringBuilder();
		json.AppendLine("{");
		var allSettings = _globalSettingRepo.GetGlobalSettings().ToArray();
		foreach (var setting in allSettings)
		{
			json.Append($"\"{setting.Id}\" : ");
			json.Append(IsSurroundWithJson(setting) ? $"{setting.Value}" : $"\"{setting.Value}\"");
			if (setting != allSettings.Last())
			{
				json.Append(",");
			}
		}
		json.AppendLine("}");
		return Deserialize(json);
	}

	private static T Deserialize(StringBuilder json)
	{
		var serializeOptions = new JsonSerializerOptions
		{
			WriteIndented = true,
			Converters =
			{
				new DelimitedStringJsonConverter(),
				new BooleanStringJsonConverter()
			}
		};
		var settingConfig = JsonSerializer.Deserialize<T>(json.ToString(), serializeOptions);
		return settingConfig ?? new T();
	}

	private static bool IsSurroundWithJson(GlobalSetting setting)
	{
		if (setting.Value.StartsWith("[") && setting.Value.EndsWith("]"))
		{
			return true;
		}
		return setting.Value.StartsWith("{") && setting.Value.EndsWith("}");
	}
}