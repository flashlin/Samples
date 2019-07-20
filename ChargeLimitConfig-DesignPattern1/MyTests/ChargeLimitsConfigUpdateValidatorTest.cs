using System;
using System.Collections.Generic;
using ChargeLimitConfig_DesignPattern1;
using Xunit;

namespace MyTests
{
	public class ChargeLimitsConfigUpdateValidatorTest
	{
		private ChargeLimitsConfig _newChargeLimitsConfig;
		private ChargeLimitsConfig _oldChargeLimitsConfig;
		private readonly ChargeLimitsConfigUpdateValidator _updateValidator;

		public ChargeLimitsConfigUpdateValidatorTest()
		{
			_updateValidator = new ChargeLimitsConfigUpdateValidator();
		}

		[Fact]
		public void Player_Increase_ChargeLimit_In24Hr()
		{
			GiveOldChargeLimitsConfig("2019/01/01", new[] {
				new ChargeLimit { PeriodDays = 1, Amount = 100 },
				new ChargeLimit { PeriodDays = 7, Amount = 100 },
				new ChargeLimit { PeriodDays = 30, Amount = 100 }
			});

			GiveNewChargeLimitsConfig("2019/01/01 10:00", new[] {
				new ChargeLimit { PeriodDays = 1, Amount = 100 },
				new ChargeLimit { PeriodDays = 7, Amount = 200 },
				new ChargeLimit { PeriodDays = 30, Amount = 100 }
			});

			ValidateShouldBe(false);
		}

		[Fact]
		public void Player_Change_ChargeLimit_To_Unlimit_In24hr()
		{
			GiveOldChargeLimitsConfig("2019/01/01", new[] {
				new ChargeLimit { PeriodDays = 1, Amount = 100 },
				new ChargeLimit { PeriodDays = 7, Amount = 100 },
				new ChargeLimit { PeriodDays = 30, Amount = 100 }
			});

			GiveNewChargeLimitsConfig("2019/01/01 10:00", new[] {
				new ChargeLimit { PeriodDays = 1, Amount = 100 },
				new ChargeLimit { PeriodDays = 7, Amount = ChargeLimit.UnlimitAmount },
				new ChargeLimit { PeriodDays = 30, Amount = 100 }
			});

			ValidateShouldBe(false);
		}
		
		[Fact]
		public void Player_Change_ChargeUnLimit_To_Limit_In24hr()
		{
			GiveOldChargeLimitsConfig("2019/01/01", new[] {
				new ChargeLimit { PeriodDays = 1, Amount = 100 },
				new ChargeLimit { PeriodDays = 7, Amount = ChargeLimit.UnlimitAmount },
				new ChargeLimit { PeriodDays = 30, Amount = 100 }
			});

			GiveNewChargeLimitsConfig("2019/01/01 10:00", new[] {
				new ChargeLimit { PeriodDays = 1, Amount = 100 },
				new ChargeLimit { PeriodDays = 7, Amount = 900 },
				new ChargeLimit { PeriodDays = 30, Amount = 100 }
			});

			ValidateShouldBe(true);
		}
		


		private static ChargeLimitsConfig CreateChargeLimits(string modifiedTime, params ChargeLimit[] chargeLimits)
		{
			var chargeLimitsConfig = new ChargeLimitsConfig()
			{
				LastModifiedTime = DateTime.Parse(modifiedTime)
			};

			foreach (var chargeLimit in chargeLimits)
			{
				chargeLimitsConfig.PeriodDayLimits.Add(chargeLimit.PeriodDays, chargeLimit);
			}

			return chargeLimitsConfig;
		}

		private void GiveNewChargeLimitsConfig(string modifiedTime, ChargeLimit[] chargeLimits)
		{
			_newChargeLimitsConfig = CreateChargeLimits(modifiedTime,
				chargeLimits);
		}

		private void GiveOldChargeLimitsConfig(string lastModifiedTime, ChargeLimit[] chargeLimits)
		{
			_oldChargeLimitsConfig = CreateChargeLimits(lastModifiedTime,
				chargeLimits);
		}

		private void ValidateShouldBe(bool expected)
		{
			var actual = _updateValidator.Validate(_oldChargeLimitsConfig, _newChargeLimitsConfig);
			Assert.Equal(expected, actual);
		}
	}
}
