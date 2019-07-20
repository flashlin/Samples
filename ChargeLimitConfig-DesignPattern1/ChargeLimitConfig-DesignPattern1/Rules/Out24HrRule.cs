using T1.Standard.DesignPatterns;

namespace ChargeLimitConfig_DesignPattern1.Rules
{
	public class Out24HrRule : ChainOfResponsibilityHandler<ValidateChargeLimitArgs>, IChargeLimitUpdateRule
	{
		public override void Handle(ValidateChargeLimitArgs args)
		{
			if (!TimeHelper.IsIn24Hr(args.LastModifiedTime, args.ModifyTime))
			{
				//implement
			}
			base.Handle(args);
		}
	}
}