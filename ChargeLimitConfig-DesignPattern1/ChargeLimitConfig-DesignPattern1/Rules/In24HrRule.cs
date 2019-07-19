using T1.Standard.DesignPatterns;

namespace ChargeLimitConfig_DesignPattern1.Rules
{
	public class In24HrRule : ChainOfResponsibilityHandler<ValidateChargeLimitArgs>, IChargeLimitUpdateRule
	{
		public override void Handle(ValidateChargeLimitArgs args)
		{
			if (!TimeHelper.IsIn24Hr(args.LastModifiedTime, args.ModifyTime))
			{
				base.Handle(args);
				return;
			}

			new CannotIncreaseRule().Handle(args);

			new CannotLimitToUnlimitRule().Handle(args);
		}
	}
}