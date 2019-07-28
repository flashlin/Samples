namespace ChargeLimitConfig_DesignPattern1.Rules
{
	public interface IChargeLimitUpdateRule
	{
		void Handle(ValidateChargeLimitArgs args);
	}
}